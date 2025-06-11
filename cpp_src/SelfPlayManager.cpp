// file: cpp_src/SelfPlayManager.cpp (Batched MCTS Final Version)
#include <pybind11/stl.h>
#include <memory>
#include <mutex>
#include <thread>
#include "SelfPlayManager.h"
#include "Gomoku.h"
#include "Node.h"
#include "SafeQueue.h"
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <atomic>

std::mutex g_io_mutex;
std::atomic<long long> g_request_id_counter(0);

// 这是Python调用的顶层函数，实现了创建管理线程、释放GIL并等待的健壮模式
void run_parallel_self_play(py::object job_queue, py::object result_queue, py::object final_data_queue, py::dict args) {
    auto manager = std::make_shared<SelfPlayManager>(job_queue, result_queue, final_data_queue, args);

    std::thread cpp_manager_thread([manager]() {
        manager->run();
    });

    {
        py::gil_scoped_release release;
        if (cpp_manager_thread.joinable()) {
            cpp_manager_thread.join();
        }
    }
}

// 构造函数保持不变
SelfPlayManager::SelfPlayManager(py::object job_queue, py::object result_queue, py::object final_data_queue, py::dict args)
    : job_queue_(job_queue), result_queue_(result_queue),
      final_data_queue_(final_data_queue), args_(args) {
    num_total_games_ = args_["num_selfPlay_episodes"].cast<int>();
    num_workers_ = args_["num_cpu_threads"].cast<int>();
    try {
        py::gil_scoped_acquire acquire;
        py::module_ queue_module = py::module_::import("queue");
        queue_empty_exc_ = queue_module.attr("Empty");
    } catch (const py::error_already_set& e) {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cerr << "Failed to import queue module or get Empty exception: " << e.what() << std::endl;
        throw;
    }
}

// run方法现在在独立的C++管理线程中执行，不需要自己管理GIL
void SelfPlayManager::run() {
    for (int i = 0; i < num_total_games_; ++i) {
        task_queue_.push(i);
    }

    {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[C++ Manager] Starting " << num_workers_ << " worker threads to play " << num_total_games_ << " games..." << std::endl;
    }

    threads_.reserve(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
        threads_.emplace_back(&SelfPlayManager::worker_func, this, i);
    }

    for (auto& t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[C++ Manager] All self-play games finished." << std::endl;
    }
}

// worker_func被彻底重构，以实现批处理MCTS
void SelfPlayManager::worker_func(int worker_id) {
    while(true) {
        int game_idx;
        if (!task_queue_.try_pop(game_idx)) {
            break;
        }

        try {
            Gomoku game;
            std::vector<std::tuple<std::vector<float>, std::vector<float>, int>> episode_data;
            const int num_simulations = args_["num_searches"].cast<int>();
            const int action_size = game.get_board_size() * game.get_board_size();

            // 单局游戏的主循环
            while (true) {
                     auto root = std::make_unique<Node>(game);

                     // 1. 搜索阶段 (CPU密集)
                     std::vector<Node*> leaves;
                     leaves.reserve(num_simulations);
                     for (int i = 0; i < num_simulations; ++i) {
                         Node* node = root.get();
                         while (node->is_fully_expanded()) {
                             node = node->select_child();
                         }

                         auto [end_value, is_terminal] = node->game_state_.get_game_ended();
                         if (is_terminal) {
                             node->backpropagate(end_value);
                         } else {
                             leaves.push_back(node);
                         }
                     }

                     // 2. 批处理评估阶段
                     if (!leaves.empty()) {
                         std::vector<std::vector<float>> state_batch;
                         state_batch.reserve(leaves.size());
                         for (const auto* leaf : leaves) {
                             state_batch.push_back(leaf->game_state_.get_state());
                         }

                         py::list policy_batch;
                         std::vector<double> value_batch;

                         long long request_id = g_request_id_counter++;
                         {
                             py::gil_scoped_acquire acquire;
                             job_queue_.attr("put")(py::make_tuple(request_id, py::cast(state_batch)));
                         }

                         bool inference_done = false;
                         while (!inference_done) {
                             {
                                 py::gil_scoped_acquire acquire;
                                 try {
                                     py::tuple result = result_queue_.attr("get_nowait")();
                                     if (result[0].cast<long long>() == request_id) {
                                         policy_batch = result[1].cast<py::list>();
                                         value_batch = result[2].cast<std::vector<double>>();
                                         inference_done = true;
                                     } else {
                                         result_queue_.attr("put")(result);
                                     }
                                 } catch (const py::error_already_set& e) {
                                     if (!e.matches(queue_empty_exc_)) {
                                          std::lock_guard<std::mutex> lock(g_io_mutex);
                                          std::cerr << "[C++ Worker " << worker_id << "] Polling error: " << e.what() << std::endl;
                                     }
                                 }
                             }
                             if (!inference_done) {
                                 std::this_thread::sleep_for(std::chrono::milliseconds(1));
                             }
                         }

                         // 3. 结果应用阶段
                         for (size_t i = 0; i < leaves.size(); ++i) {
                             Node* leaf = leaves[i];
                             std::vector<float> policy;
                             py::list py_policy = policy_batch[i].cast<py::list>();
                             policy.reserve(py::len(py_policy));
                             for(py::handle item : py_policy) {
                                 policy.push_back(item.cast<float>());
                             }
                             double value = value_batch[i];
                             leaf->expand(policy);
                             leaf->backpropagate(value);
                         }
                     }

                     // 4. 选择并执行动作
                     std::vector<float> action_probs(action_size, 0.0f);
                     for (const auto& child : root->children_) {
                         if(child && child->action_taken_ >= 0 && child->action_taken_ < action_size)
                             action_probs[child->action_taken_] = static_cast<float>(child->visit_count_);
                     }

                     float sum_visits = std::accumulate(action_probs.begin(), action_probs.end(), 0.0f);
                     if (sum_visits > 0) {
                         for (float& p : action_probs) { p /= sum_visits; }
                     }
                     episode_data.emplace_back(root->game_state_.get_state(), action_probs, game.get_current_player());

                     int action = -1;
                     float max_visits_count = -1.0f;
                     for (size_t i = 0; i < action_probs.size(); ++i) {
                         if (action_probs[i] > max_visits_count) {
                             max_visits_count = action_probs[i];
                             action = static_cast<int>(i);
                         }
                     }

                     if (action == -1) {
                         auto valid_moves = game.get_valid_moves();
                         for (size_t i = 0; i < valid_moves.size(); ++i) {
                             if (valid_moves[i]) {
                                 action = static_cast<int>(i);
                                 break;
                             }
                         }
                         if (action == -1) {
                             {
                                 std::lock_guard<std::mutex> lock(g_io_mutex);
                                 std::cout << "[C++ Worker " << worker_id << "] No valid moves left. Ending game prematurely." << std::endl;
                             }
                             break;
                         }
                     }

                     game.execute_move(action);

                     {
                         std::lock_guard<std::mutex> lock(g_io_mutex);
                         // 打印一个清晰的分隔符，包含步数、执行的玩家和动作
                         std::cout << "\n=============== Game " << game_idx << ", Move " << game.get_move_number()
                                   << " by player " << (game.get_current_player() * -1) // 乘以-1得到刚刚走棋的玩家
                                   << " (action: " << action << ") ===============\n";
                         game.print_board(); // 调用我们写好的打印函数
                         std::cout << "=========================================================\n" << std::endl;
                     }

                     // 在 worker_func 函数的游戏主循环 while(true) 的末尾...

                     auto [final_value, is_done] = game.get_game_ended();

                     // ==================== 从这里开始替换 ====================
                     if (is_done) {
                         // 步骤1：在纯C++的世界里，用C++的类型准备好所有训练数据
                         std::vector<std::tuple<std::vector<float>, std::vector<float>, double>> cpp_training_examples;
                         cpp_training_examples.reserve(episode_data.size());

                         for (const auto& example : episode_data) {
                             // 使用我们之前修正过的、正确的价值计算逻辑
                             double corrected_value = final_value * std::get<2>(example);
                             cpp_training_examples.emplace_back(std::get<0>(example), std::get<1>(example), corrected_value);
                         }

                         // 步骤2：获取GIL，然后一次性完成所有与Python的交互
                         {
                             py::gil_scoped_acquire acquire;

                             // 将C++的vector一次性转换成Python的list
                             py::list training_examples_list;
                             for (const auto& ex : cpp_training_examples) {
                                 training_examples_list.append(py::make_tuple(
                                     py::cast(std::get<0>(ex)),
                                     py::cast(std::get<1>(ex)),
                                     py::cast(std::get<2>(ex))
                                 ));
                             }

                             // 创建要发送的Python字典
                             py::dict data_to_send;
                             data_to_send["type"] = "data";
                             data_to_send["data"] = training_examples_list;

                             // 将最终的Python对象放入队列
                             final_data_queue_.attr("put")(data_to_send);
                         } // GIL在此处被安全释放

                         break; // 结束当前这局游戏的循环
                     }
                     // ==================== 到这里替换结束 ====================
                 } // while(true) for a single game ends here
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cerr << "[C++ Worker " << worker_id << ", Game " << game_idx << "] FATAL_ERROR: A C++ standard exception occurred! what(): " << e.what() << std::endl;
        } catch (...) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cerr << "[C++ Worker " << worker_id << ", Game " << game_idx << "] FATAL_ERROR: An unknown exception occurred!" << std::endl;
        }
    }
}

// play_game_for_eval 函数保持不变
int play_game_for_eval(py::object model1, py::object model2, py::dict args) {
    // ... 此函数无需改动 ...
    Gomoku game;
    std::map<int, py::object> models = {{1, model1}, {-1, model2}};
    const int num_simulations = args["num_eval_simulations"].cast<int>();
    const int board_size = args["board_size"].cast<int>();
    while(true) {
        py::object current_model = models.at(game.get_current_player());
        auto root = std::make_unique<Node>(game);
        for (int i = 0; i < num_simulations; ++i) {
            Node* node = root.get();
            while (node->is_fully_expanded()) {
                node = node->select_child();
            }
            auto [end_value, is_terminal] = node->game_state_.get_game_ended();
            double value = end_value;
            if (!is_terminal) {
                py::tuple result;
                {
                    py::gil_scoped_acquire acquire;
                    auto state = node->game_state_.get_state();
                    auto state_tensor = py::module_::import("torch").attr("tensor")(py::cast(state))
                                          .attr("unsqueeze")(0)
                                          .attr("to")("cpu", py::module_::import("torch").attr("float32"))
                                          .attr("view")(-1, 6, board_size, board_size);
                    result = current_model.attr("forward")(state_tensor);
                }
                std::vector<float> policy;
                {
                    py::gil_scoped_acquire acquire;
                    py::list py_policy = result[0].attr("exp")().attr("squeeze")(0).attr("cpu")().attr("tolist")();
                    policy.reserve(py::len(py_policy));
                    for(py::handle h : py_policy) {
                        policy.push_back(h.cast<float>());
                    }
                }
                value = result[1].attr("item")().cast<double>();
                node->expand(policy);
            }
            node->backpropagate(value);
        }
        int action = -1;
        int max_visits = -1;
        for(const auto& child : root->children_){
            if(child->visit_count_ > max_visits){
                max_visits = child->visit_count_;
                action = child->action_taken_;
            }
        }
        if (action == -1) {
            auto valid_moves = game.get_valid_moves();
            for(size_t i = 0; i < valid_moves.size(); ++i){
                if(valid_moves[i]) {
                    action = i;
                    break;
                }
            }
        }
        game.execute_move(action);
        auto [final_value, is_done] = game.get_game_ended();
        if(is_done){
            if(std::abs(final_value) < 0.01) return 0;
            return static_cast<int>(final_value * game.get_current_player());
        }
    }
}