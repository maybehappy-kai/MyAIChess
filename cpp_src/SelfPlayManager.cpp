// file: cpp_src/SelfPlayManager.cpp (完全修正版)
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

// Python调用的顶层函数
void run_parallel_self_play(const std::string& model_path, bool use_gpu, py::object final_data_queue, py::dict args) {
    auto engine = std::make_shared<InferenceEngine>(model_path, use_gpu);
    auto manager = std::make_shared<SelfPlayManager>(engine, final_data_queue, args);

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

// ====================== 核心修正区域：构造函数 ======================
// 在构造函数中，从Python字典中提取所需参数并用C++成员变量存储
SelfPlayManager::SelfPlayManager(std::shared_ptr<InferenceEngine> engine, py::object final_data_queue, py::dict args)
    : engine_(engine), final_data_queue_(final_data_queue) {

    this->num_total_games_ = args["num_selfPlay_episodes"].cast<int>();
    this->num_workers_ = args["num_cpu_threads"].cast<int>();
    this->num_simulations_ = args["num_searches"].cast<int>();
}
// ===============================================================

void SelfPlayManager::run() {
    for (int i = 0; i < this->num_total_games_; ++i) {
        task_queue_.push(i);
    }

    {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[C++ Manager] Starting " << this->num_workers_ << " worker threads to play " << this->num_total_games_ << " games..." << std::endl;
    }

    threads_.reserve(this->num_workers_);
    for (int i = 0; i < this->num_workers_; ++i) {
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

void SelfPlayManager::worker_func(int worker_id) {
    while (true) {
        int game_idx;
        if (!task_queue_.try_pop(game_idx)) {
            break;
        }

        try {
            Gomoku game;
            std::vector<std::tuple<std::vector<float>, std::vector<float>, int>> episode_data;

            // ====================== 核心修正区域：工作函数 ======================
            // 直接使用C++成员变量，不再访问Python字典
            const int num_simulations = this->num_simulations_;
            // ===============================================================

            const int action_size = game.get_board_size() * game.get_board_size();

            while (true) {
                auto root = std::make_unique<Node>(game);

                std::vector<Node*> leaves;
                leaves.reserve(num_simulations);
                for (int i = 0; i < num_simulations; ++i) {
                    Node* node = root.get();
                    while (node->is_fully_expanded()) {
                        node = node->select_child();
                    }
                    auto [end_value, is_terminal] = node->game_state_.get_game_ended();
                    double value_to_propagate = end_value;
                    if (is_terminal) {
                        value_to_propagate *= node->game_state_.get_current_player();
                        node->backpropagate(value_to_propagate);
                        continue;
                    }
                    leaves.push_back(node);
                }

                if (!leaves.empty()) {
                    std::sort(leaves.begin(), leaves.end());
                    leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
                    std::vector<std::vector<float>> state_batch;
                    state_batch.reserve(leaves.size());
                    for (const auto* leaf : leaves) {
                        state_batch.push_back(leaf->game_state_.get_state());
                    }
                    auto [policy_batch, value_batch] = engine_->infer(state_batch);
                    for (size_t i = 0; i < leaves.size(); ++i) {
                        Node* leaf = leaves[i];
                        leaf->expand(policy_batch[i]);
                        leaf->backpropagate(static_cast<double>(value_batch[i]));
                    }
                }

                std::vector<float> action_probs(action_size, 0.0f);
                if (!root->children_.empty()) {
                    for (const auto& child : root->children_) {
                        if (child && child->action_taken_ >= 0 && child->action_taken_ < action_size) {
                            action_probs[child->action_taken_] = static_cast<float>(child->visit_count_);
                        }
                    }
                    float sum_visits = std::accumulate(action_probs.begin(), action_probs.end(), 0.0f);
                    if (sum_visits > 0) {
                        for (float& p : action_probs) { p /= sum_visits; }
                    }
                }
                episode_data.emplace_back(root->game_state_.get_state(), action_probs, game.get_current_player());

                int action = -1;
                if (!root->children_.empty()) {
                    int max_visits = -1;
                    for (const auto& child : root->children_) {
                        if (child && child->visit_count_ > max_visits) {
                            max_visits = child->visit_count_;
                            action = child->action_taken_;
                        }
                    }
                }
                if (action == -1) {
                    auto valid_moves = game.get_valid_moves();
                    std::vector<int> valid_move_indices;
                    for (size_t i = 0; i < valid_moves.size(); ++i) {
                        if (valid_moves[i]) {
                            valid_move_indices.push_back(i);
                        }
                    }
                    if (!valid_move_indices.empty()) {
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_int_distribution<> distrib(0, valid_move_indices.size() - 1);
                        action = valid_move_indices[distrib(gen)];
                    } else {
                        break;
                    }
                }
                game.execute_move(action);

                auto [final_value, is_done] = game.get_game_ended();
                if (is_done) {
                    std::vector<std::tuple<std::vector<float>, std::vector<float>, double>> cpp_training_examples;
                    cpp_training_examples.reserve(episode_data.size());
                    for (const auto& example : episode_data) {
                        double corrected_value = final_value * std::get<2>(example);
                        cpp_training_examples.emplace_back(std::get<0>(example), std::get<1>(example), corrected_value);
                    }
                    {
                        py::gil_scoped_acquire acquire;
                        py::list training_examples_list;
                        for (const auto& ex : cpp_training_examples) {
                            training_examples_list.append(py::make_tuple(
                                py::cast(std::get<0>(ex)),
                                py::cast(std::get<1>(ex)),
                                py::cast(std::get<2>(ex))
                            ));
                        }
                        py::dict data_to_send;
                        data_to_send["type"] = "data";
                        data_to_send["data"] = training_examples_list;
                        final_data_queue_.attr("put")(data_to_send);
                    }
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

// 评估函数 play_game_for_eval 保持不变
int play_game_for_eval(py::object model1, py::object model2, py::dict args) {
    // ... 此函数无需改动，代码省略 ...
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