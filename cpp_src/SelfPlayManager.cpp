// file: cpp_src/SelfPlayManager.cpp (Thread Pool Final Version)
#include <pybind11/stl.h>
#include <memory>
#include <mutex>
#include "SelfPlayManager.h"
#include "Gomoku.h"
#include "Node.h"
#include "SafeQueue.h" // 引入我们之前创建的线程安全队列
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

// 文件: cpp_src/SelfPlayManager.cpp

void run_parallel_self_play(py::object job_queue, py::object result_queue, py::object final_data_queue, py::dict args) {

    // ==================== 从这里开始替换 ====================

    // 1. 在GIL被持有时，安全地创建Manager对象。
    // 使用智能指针，方便管理其生命周期。
    auto manager = std::make_shared<SelfPlayManager>(job_queue, result_queue, final_data_queue, args);

    // 2. 创建一个新的C++“管理线程”，它的唯一任务就是调用 manager->run()
    std::thread cpp_manager_thread([manager]() {
        manager->run();
    });

    // 3. Python主线程现在可以安全地释放GIL，并等待C++管理线程完成所有工作
    {
        py::gil_scoped_release release;
        if (cpp_manager_thread.joinable()) {
            cpp_manager_thread.join();
        }
    }

    // ==================== 到这里替换结束 ====================
}

SelfPlayManager::SelfPlayManager(py::object job_queue, py::object result_queue, py::object final_data_queue, py::dict args)
    : job_queue_(job_queue), result_queue_(result_queue),
      final_data_queue_(final_data_queue), args_(args) {
    num_total_games_ = args_["num_selfPlay_episodes"].cast<int>();
    num_workers_ = args_["num_cpu_threads"].cast<int>(); // 使用新的参数来确定线程数
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

// 【新】run方法：实现线程池逻辑
void SelfPlayManager::run() {
    // 1. 将所有游戏任务放入任务队列
    for (int i = 0; i < num_total_games_; ++i) {
        task_queue_.push(i);
    }

    {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[C++ Manager] Starting " << num_workers_ << " worker threads to play " << num_total_games_ << " games..." << std::endl;
    }

    // 2. 启动固定数量的worker线程
    threads_.reserve(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
        threads_.emplace_back(&SelfPlayManager::worker_func, this, i);
    }

    // 3. 等待所有worker线程完成它们的工作
    {
        for (auto& t : threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[C++ Manager] All worker threads finished." << std::endl;
    }
}

// 【新】worker_func方法：实现从任务队列取活干的逻辑
void SelfPlayManager::worker_func(int worker_id) {
    // 每个worker在一个循环中不断地从任务队列获取游戏来玩
    while(true) {
        int game_idx;
        // 尝试从任务队列中取出一个任务
        if (!task_queue_.try_pop(game_idx)) {
            // 如果队列为空，说明所有游戏都已被分配，线程可以退出
            break;
        }

        try {
            // ... 这部分是完整的单局游戏逻辑，与我们之前的最终版几乎一样 ...
            Gomoku game;
            std::vector<std::tuple<std::vector<float>, std::vector<float>, int>> episode_data;
            const int num_simulations = args_["num_searches"].cast<int>();
            const int board_size = game.get_board_size();
            const int action_size = board_size * board_size;
            while (true) {
                auto root = std::make_unique<Node>(game);
                for (int i = 0; i < num_simulations; ++i) {
                    Node* node = root.get();
                    while (node->is_fully_expanded()) {
                        node = node->select_child();
                    }
                    auto [end_value, is_terminal] = node->game_state_.get_game_ended();
                    double value = end_value;
                    if (!is_terminal) {
                        long long request_id = g_request_id_counter++;
                        std::vector<float> state = node->game_state_.get_state();
                        {
                            py::gil_scoped_acquire acquire;
                            job_queue_.attr("put")(py::make_tuple(request_id, py::cast(state)));
                        }
                        bool inference_done = false;
                        // 使用一个布尔标志来控制循环，而不是无限循环
                        while (!inference_done) {
                            { // 创建一个新的、独立的作用域，专门用于管理GIL

                                // 1. 在进入作用域时，自动获取GIL
                                py::gil_scoped_acquire acquire;

                                try {
                                    // 2. 所有与Python对象（队列）的交互都在这个受保护的作用域内进行
                                    py::tuple result = result_queue_.attr("get_nowait")();

                                    // 3. 检查返回的结果ID是否是当前线程正在等待的那个
                                    if (result[0].cast<long long>() == request_id) {
                                        // 如果ID匹配，处理结果

                                        // 从Python的list安全地转换为C++的vector
                                        std::vector<float> policy;
                                        py::list py_policy = result[1].cast<py::list>();
                                        policy.reserve(py::len(py_policy));
                                        for (py::handle item : py_policy) {
                                            policy.push_back(item.cast<float>());
                                        }

                                        // 获取value值
                                        value = result[2].cast<double>();

                                        // 使用获取到的策略来扩展MCTS节点
                                        node->expand(policy);

                                        // 标记任务完成，以便退出外层的while循环
                                        inference_done = true;

                                    } else {
                                        // 如果ID不匹配，说明是其他线程的结果，把它放回队列
                                        result_queue_.attr("put")(result);
                                    }
                                } catch (const py::error_already_set& e) {
                                    // 如果捕获到异常，检查它是否是“队列为空”。
                                    // 这是轮询中的正常情况，我们忽略它，稍后休眠。
                                    if (!e.matches(queue_empty_exc_)) {
                                        // 如果是其他未知异常，则打印出来方便调试
                                         std::lock_guard<std::mutex> lock(g_io_mutex);
                                         std::cerr << "[C++ Worker " << worker_id << "] Polling error: " << e.what() << std::endl;
                                    }
                                }

                            } // 4. 当代码执行到这个右花括号时，`acquire`对象被销毁，GIL被自动、安全地释放

                            // 5. 如果推理尚未完成，就在不持有GIL的情况下让线程休眠
                            if (!inference_done) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                        }
                    }
                    node->backpropagate(value);
                }
                std::vector<float> action_probs(action_size, 0.0f);
                for (const auto& child : root->children_) {
                    if(child && child->action_taken_ >= 0 && child->action_taken_ < action_size)
                        action_probs[child->action_taken_] = static_cast<float>(child->visit_count_);
                }
                float sum_visits = std::accumulate(action_probs.begin(), action_probs.end(), 0.0f);
                if (sum_visits > 0) { for (float& p : action_probs) { p /= sum_visits; } }
                episode_data.emplace_back(root->game_state_.get_state(), action_probs, root->game_state_.get_current_player());
                int action = std::distance(action_probs.begin(), std::max_element(action_probs.begin(), action_probs.end()));
                game.execute_move(action);
                auto [final_value, is_done] = game.get_game_ended();
                if (is_done) {
                    py::gil_scoped_acquire acquire;
                    py::list training_examples;
                    for (const auto& example : episode_data) {
                        double corrected_value = final_value * std::get<2>(example);
                        training_examples.append(py::make_tuple(std::get<0>(example), std::get<1>(example), corrected_value));
                    }
                    py::dict data_to_send;
                    data_to_send["type"] = "data";
                    data_to_send["data"] = training_examples;
                    final_data_queue_.attr("put")(data_to_send);
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cerr << "[C++ Worker " << worker_id << ", Game " << game_idx << "] FATAL_ERROR: A C++ standard exception occurred! what(): " << e.what() << std::endl;
        } catch (...) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cerr << "[C++ Worker " << worker_id << ", Game " << game_idx << "] FATAL_ERROR: An unknown exception occurred!" << std::endl;
        }
    }
}

int play_game_for_eval(py::object model1, py::object model2, py::dict args) {
    Gomoku game;
    std::map<int, py::object> models = {{1, model1}, {-1, model2}};
    // 修正：从args字典中读取board_size和num_eval_simulations
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