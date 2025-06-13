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

// ====================== 新增：高效并行评估的完整实现 ======================

// C++评估任务的顶层入口
py::dict run_parallel_evaluation(const std::string& model1_path, const std::string& model2_path, bool use_gpu, py::dict args, int mode) { // <-- 增加mode参数
    auto engine1 = std::make_shared<InferenceEngine>(model1_path, use_gpu);
    auto engine2 = std::make_shared<InferenceEngine>(model2_path, use_gpu);

    auto eval_manager = std::make_shared<EvaluationManager>(engine1, engine2, args, mode); // <-- 将mode传给构造函数

    std::thread cpp_eval_thread([eval_manager]() {
        eval_manager->run();
    });

    {
        py::gil_scoped_release release;
        if (cpp_eval_thread.joinable()) {
            cpp_eval_thread.join();
        }
    }

    return eval_manager->get_results();
}

// EvaluationManager 类的实现
EvaluationManager::EvaluationManager(std::shared_ptr<InferenceEngine> engine1, std::shared_ptr<InferenceEngine> engine2, py::dict args, int mode) // <-- 增加mode参数
    : engine1_(engine1), engine2_(engine2), evaluation_mode_(mode) { // <-- 保存mode
    num_total_games_ = args["num_eval_games"].cast<int>();
    num_workers_ = args["num_cpu_threads"].cast<int>();
    num_simulations_ = args["num_eval_simulations"].cast<int>();
    scores_[1] = 0;
    scores_[-1] = 0;
    scores_[0] = 0;
}

py::dict EvaluationManager::get_results() const {
    py::dict results;
    results["model1_wins"] = scores_.at(1);
    results["model2_wins"] = scores_.at(-1);
    results["draws"] = scores_.at(0);
    return results;
}

void EvaluationManager::run() {
    for (int i = 0; i < num_total_games_; ++i) {
        task_queue_.push(i);
    }
    threads_.reserve(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
        threads_.emplace_back(&EvaluationManager::worker_func, this, i);
    }
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

// 评估工作线程，核心逻辑在此
void EvaluationManager::worker_func(int worker_id) {
    while (true) {
        int game_idx;
        if (!task_queue_.try_pop(game_idx)) {
            break;
        }

        Gomoku game;
        // ====================== 核心逻辑修改 ======================
                auto& p1_engine = engine1_; // Model 1 默认执黑 (Player 1)
                auto& p2_engine = engine2_; // Model 2 默认执白 (Player 2)

                if (evaluation_mode_ == 0) { // Mode 0: 交替先后手
                    bool swap_models = (game_idx % 2 != 0);
                    if(swap_models) {
                        p1_engine = engine2_;
                        p2_engine = engine1_;
                    }
                } else if (evaluation_mode_ == 1) { // Mode 1: 固定 Model 1 先手
                    // 不需要做任何事，p1_engine已经是engine1_
                } else if (evaluation_mode_ == 2) { // Mode 2: 固定 Model 2 先手
                    p1_engine = engine2_;
                    p2_engine = engine1_;
                }
                // ========================================================

        std::map<int, std::shared_ptr<InferenceEngine>> models = {
            {1, p1_engine},
            {-1, p2_engine}
        };

        while (true) {
            int current_player = game.get_current_player();
            auto& current_engine = models.at(current_player);

            auto root = std::make_unique<Node>(game);

            // MCTS 搜索过程 (与自对弈类似，但使用C++推理引擎)
            std::vector<Node*> leaves;
            leaves.reserve(num_simulations_);
            for (int i = 0; i < num_simulations_; ++i) {
                Node* node = root.get();
                while (node->is_fully_expanded()) node = node->select_child();
                auto [end_value, is_terminal] = node->game_state_.get_game_ended();
                if (is_terminal) {
                    node->backpropagate(end_value * node->game_state_.get_current_player());
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

                // 使用当前玩家对应的C++推理引擎
                auto [policy_batch, value_batch] = current_engine->infer(state_batch);

                for (size_t i = 0; i < leaves.size(); ++i) {
                    leaves[i]->expand(policy_batch[i]);
                    leaves[i]->backpropagate(static_cast<double>(value_batch[i]));
                }
            }

            // 选择动作 (确定性选择)
            int action = -1;
            int max_visits = -1;
            for (const auto& child : root->children_) {
                if (child && child->visit_count_ > max_visits) {
                    max_visits = child->visit_count_;
                    action = child->action_taken_;
                }
            }
             if (action == -1) { // Fallback for no moves
                auto valid_moves = game.get_valid_moves();
                for (size_t i = 0; i < valid_moves.size(); ++i) if (valid_moves[i]) { action = i; break; }
            }

            if (action == -1) break; // No moves possible, end as draw

            game.execute_move(action);

            // 游戏结束判断
            auto [final_value, is_done] = game.get_game_ended();
            if (is_done) {
                int winner_code = 0; // 默认为平局
                if (std::abs(final_value) > 0.01) { // 判断是否真的分出了胜负
                    int winner_result = static_cast<int>(final_value); // 1 代表P1胜, -1 代表P2胜

                    // 关键：我们要记录的是 model1 和 model2 的胜负
                    // winner_code = 1 代表 model1 胜, -1 代表 model2 胜
                    if (winner_result == 1) { // 如果 P1 胜了
                        // 检查P1是哪个模型
                        winner_code = (&p1_engine == &engine1_) ? 1 : -1;
                    } else { // 如果 P2 胜了
                        // 检查P2是哪个模型
                        winner_code = (&p2_engine == &engine1_) ? 1 : -1;
                    }
                }

                            // 使用线程锁，安全地更新总计分板
                            {
                                std::lock_guard<std::mutex> lock(results_mutex_);
                                scores_[winner_code]++;
                            }

                            break; // 退出当前这局游戏的循环
                            // ======================================================
            }
        }
    }
}
