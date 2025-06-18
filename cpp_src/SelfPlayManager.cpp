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

// ======================= 新增的、仅供人机对战使用的引擎缓存 =======================
// 用于存储已经加载的模型，避免在对战中重复从硬盘读取
static std::mutex g_engine_mutex;
static std::map<std::string, std::shared_ptr<InferenceEngine>> g_engines;

// 一个辅助函数，用于获取一个缓存的或新加载的引擎实例
std::shared_ptr<InferenceEngine> get_cached_engine(const std::string& model_path, bool use_gpu) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (g_engines.find(model_path) == g_engines.end()) {
        std::cout << "[C++ Engine Cache] Caching new model for interactive play: " << model_path << std::endl;
        g_engines[model_path] = std::make_shared<InferenceEngine>(model_path, use_gpu);
    }
    return g_engines[model_path];
}
// ==============================================================================

std::mutex g_io_mutex;
std::atomic<long long> g_request_id_counter(0);

// ====================== 【新增】狄利克雷噪声辅助函数 ======================
// 在文件靠前的位置 (例如，在 g_io_mutex 定义后) 添加这个函数
void add_dirichlet_noise(
    std::vector<float>& policy,
    const std::vector<bool>& valid_moves,
    double alpha,
    double epsilon)
{
    // 使用 thread_local 确保每个工作线程拥有自己独立的、线程安全的随机数生成器
    static thread_local std::mt19937 generator(std::random_device{}());

    // 统计有多少个合法走法
    int num_valid_moves = 0;
    for (bool valid : valid_moves) {
        if (valid) {
            num_valid_moves++;
        }
    }
    // 如果合法走法少于2个，添加噪声没有意义
    if (num_valid_moves <= 1) return;

    // 为每个合法走法从伽马分布中采样一个值
    std::vector<float> noise_values;
    noise_values.reserve(num_valid_moves);
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float sum_of_noise = 0.0f;

    for (size_t i = 0; i < valid_moves.size(); ++i) {
        if (valid_moves[i]) {
            float noise = gamma(generator);
            noise_values.push_back(noise);
            sum_of_noise += noise;
        }
    }

    // 归一化伽马分布的采样值，得到最终的狄利克雷噪声
    if (sum_of_noise > 0.0f) {
        for (float& n : noise_values) {
            n /= sum_of_noise;
        }
    }

    // 将噪声混合进原始策略中
    int noise_idx = 0;
    for (size_t i = 0; i < policy.size(); ++i) {
        if (valid_moves[i]) {
            policy[i] = (1.0f - epsilon) * policy[i] + epsilon * noise_values[noise_idx];
            noise_idx++;
        }
    }
}
// ===========================================================================

// ====================== 新增：活二威胁检测启发函数 ======================
void apply_threat_detection_bias(
    std::vector<float>& policy,
    const Gomoku& game_state,
    float bonus_strength)
{
    const int board_size = game_state.get_board_size();
    const int player = game_state.get_current_player();
    auto valid_moves = game_state.get_valid_moves();

    std::vector<int> threat_moves;

    for (int action = 0; action < board_size * board_size; ++action) {
        if (!valid_moves[action]) continue;

        Gomoku next_state = game_state;
        next_state.execute_move(action);

        // 在新状态下，检查对手是否已经输了 (因为我们的落子可能直接形成三连)
        // 这个检查很简单，如果对手没有合法落子点，说明我们赢了
        if (std::none_of(next_state.get_valid_moves().begin(), next_state.get_valid_moves().end(), [](bool v){ return v; })) {
             threat_moves.push_back(action);
             continue;
        }
    }

    if (!threat_moves.empty()) {
        float total_policy = 0.0f;
        // 给所有识别出的威胁点加上巨大的奖励
        for (int move : threat_moves) {
            policy[move] += bonus_strength;
        }
        // 重新归一化
        for(float p : policy) {
            if (p > 0) total_policy += p;
        }
        if (total_policy > 0) {
            for (float& p : policy) {
                if (p > 0) p /= total_policy;
            }
        }
    }
}
// =======================================================================

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
    this->dirichlet_alpha_ = args["dirichlet_alpha"].cast<double>();
        this->dirichlet_epsilon_ = args["dirichlet_epsilon"].cast<double>();
         // 【新增】从Python字典中获取温度参数
            this->temperature_start_ = args["temperature_start"].cast<double>();
            this->temperature_end_ = args["temperature_end"].cast<double>();
            this->temperature_decay_moves_ = args["temperature_decay_moves"].cast<int>();
            this->mcts_batch_size_ = args["mcts_batch_size"].cast<int>();
             this->enable_opening_bias_ = args["enable_opening_bias"].cast<bool>();
                this->opening_bias_strength_ = args["opening_bias_strength"].cast<float>();
                this->enable_threat_detection_ = args["enable_threat_detection"].cast<bool>();
                this->threat_detection_bonus_ = args["threat_detection_bonus"].cast<float>();
                this->enable_territory_heuristic_ = args["enable_territory_heuristic"].cast<bool>();
                this->territory_heuristic_weight_ = args["territory_heuristic_weight"].cast<double>();
                this->board_size_ = args.contains("board_size") ? args["board_size"].cast<int>() : 9;
                    this->num_rounds_ = args.contains("num_rounds") ? args["num_rounds"].cast<int>() : 25;
                    this->history_steps_ = args.contains("history_steps") ? args["history_steps"].cast<int>() : 0;
                    this->num_channels_ = args["num_channels"].cast<int>();
                    this->c_puct_ = args["C"].cast<double>();
}
// ===============================================================

// ==================== 新增：搬运工线程的完整实现 ====================
void SelfPlayManager::collector_func() {
    while (completed_games_count_ < num_total_games_) {
        TrainingDataPacket packet;
        if (data_collector_queue_.try_pop(packet)) {
            // 成功从中转站取出数据包
            // 现在，只有这个线程需要获取GIL来与Python交互
            py::gil_scoped_acquire acquire;

            py::list training_examples_list;
            for (const auto& ex : packet) {
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
        } else {
            // 中转站暂时为空，短暂休眠一下，避免空转浪费CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void SelfPlayManager::run() {
    for (int i = 0; i < this->num_total_games_; ++i) {
        task_queue_.push(i);
    }

    {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[C++ Manager] Starting " << this->num_workers_ << " worker threads to play " << this->num_total_games_ << " games..." << std::endl;
    }

    // ==================== 新增：启动搬运工线程 ====================
        collector_thread_ = std::thread(&SelfPlayManager::collector_func, this);

    threads_.reserve(this->num_workers_);
    for (int i = 0; i < this->num_workers_; ++i) {
        threads_.emplace_back(&SelfPlayManager::worker_func, this, i);
    }

    for (auto& t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }

    // ==================== 新增：等待搬运工线程结束 ====================
        if (collector_thread_.joinable()) {
            collector_thread_.join();
        }

    {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[C++ Manager] All self-play games finished." << std::endl;
    }
}

// ====================== 最终版工作函数 (包含正确的MCTS和所有原有功能) ======================
void SelfPlayManager::worker_func(int worker_id) {
    while (true) {
        int game_idx;
        if (!task_queue_.try_pop(game_idx)) {
            break; // 任务队列空，线程退出
        }

        try {
            Gomoku game(this->board_size_, this->num_rounds_, this->history_steps_);
            std::vector<std::tuple<std::vector<float>, std::vector<float>, int>> episode_data;
            const int action_size = game.get_board_size() * game.get_board_size();

            // ====================== 单局游戏主循环 ======================
            while (true) {
                auto initial_state_ptr = std::make_shared<const Gomoku>(game);
                auto root = std::make_unique<Node>(initial_state_ptr, nullptr, -1, 1.0f);

                // === 1. 正确的MCTS批处理搜索 ===
                const int num_simulations = this->num_simulations_;
                const int batch_size = this->mcts_batch_size_;

                for (int i = 0; i < num_simulations; i += batch_size) {
                    std::vector<Node*> leaves_batch;
                    leaves_batch.reserve(batch_size);
                    std::vector<std::vector<float>> state_batch;
                    state_batch.reserve(batch_size);

                    // 1a. 收集一个批次的叶子节点
                    for (int j = 0; j < batch_size && i + j < num_simulations; ++j) {
                        Node* node = root.get();

                        node->virtual_loss_count_++;

                        while (node->is_fully_expanded()) {
                            node = node->select_child(this->c_puct_);
                            node->virtual_loss_count_++;
                        }

                        auto [end_value, is_terminal] = node->game_state_->get_game_ended();
                        if (is_terminal) {
                            Node* temp_node = node;
                            while(temp_node != nullptr) {
                                temp_node->virtual_loss_count_--;
                                temp_node = temp_node->parent_;
                            }
                            double value_to_propagate = end_value * node->game_state_->get_current_player();
                            node->backpropagate(value_to_propagate);
                        } else {
                            leaves_batch.push_back(node);
                            state_batch.push_back(node->game_state_->get_state());
                        }
                    }

                    if (leaves_batch.empty()) {
                        continue;
                    }

                    // 1b. 对批次进行神经网络推理
                    auto [policy_batch, value_batch] = engine_->infer(state_batch, this->board_size_, this->num_channels_);

                    // 1c. 扩展和反向传播批次中的每一个叶子节点
                    for (size_t k = 0; k < leaves_batch.size(); ++k) {
                        Node* leaf = leaves_batch[k];
                        auto current_policy = policy_batch[k];

                        // ====================== 在此注入您的完整逻辑 ======================
                        if (leaf->parent_ == nullptr) { // 只对根节点操作
                            // --- 注入开局偏置 ---
                            if (this->enable_opening_bias_ && leaf->game_state_->get_move_number() == 0) {
                                const int board_size = leaf->game_state_->get_board_size();
                                float bias_strength = this->opening_bias_strength_;
                                std::vector<float> opening_bias(board_size * board_size, 0.0f);
                                for (int r = 0; r < board_size; ++r) {
                                    for (int c = 0; c < board_size; ++c) {
                                        int dist_from_edge = std::min({r, c, board_size - 1 - r, board_size - 1 - c});
                                        float bias = 0.0f;
                                        if (dist_from_edge == 0) bias = 0.1f;
                                        else if (dist_from_edge == 1) bias = 0.4f;
                                        else if (dist_from_edge == 2) bias = 0.8f;
                                        else bias = 1.0f;
                                        opening_bias[r * board_size + c] = bias;
                                    }
                                }
                                float policy_sum = 0.0f;
                                for(size_t i = 0; i < current_policy.size(); ++i) {
                                    current_policy[i] += bias_strength * opening_bias[i];
                                    policy_sum += current_policy[i];
                                }
                                if (policy_sum > 0.0f) {
                                    for(size_t i = 0; i < current_policy.size(); ++i) {
                                        current_policy[i] /= policy_sum;
                                    }
                                }
                            }

                            // --- 调用威胁检测 ---
                            if (this->enable_threat_detection_) {
                                float bonus = this->threat_detection_bonus_;
                                // 注意：apply_threat_detection_bias 需要一个Gomoku对象引用，所以我们用 * 解引用指针
                                apply_threat_detection_bias(current_policy, *leaf->game_state_, bonus);
                            }

                            // --- 添加狄利克雷噪声 ---
                            add_dirichlet_noise(
                                current_policy,
                                leaf->game_state_->get_valid_moves(), // 注意这里是 ->
                                this->dirichlet_alpha_,
                                this->dirichlet_epsilon_
                            );
                        }
                        // ====================== 注入逻辑结束 ======================

                        // 【统一的扩展逻辑】使用可能已被修改过的 current_policy 来创建子节点
                        const auto& parent_state = *leaf->game_state_;
                        const auto valid_moves = parent_state.get_valid_moves();
                        leaf->children_.reserve(valid_moves.size());

                        for (size_t action = 0; action < current_policy.size(); ++action) {
                            if (current_policy[action] > 0.0f && valid_moves[action]) {
                                auto next_game_state = std::make_shared<Gomoku>(parent_state);
                                next_game_state->execute_move(action);
                                leaf->children_.push_back(std::make_unique<Node>(
                                    next_game_state, leaf, action, current_policy[action]
                                ));
                            }
                        }
                        // --- 修改开始: 混合价值启发 ---
                            double nn_value = static_cast<double>(value_batch[k]);
                            double final_value = nn_value;

                            if (this->enable_territory_heuristic_) { // <--- 修改点
                                double weight = this->territory_heuristic_weight_; // <--- 修改点
                                const int board_size = leaf->game_state_->get_board_size();

                                // 计算归一化的领地分数作为启发值 (-1.0 to 1.0)
                                double territory_score = static_cast<double>(leaf->game_state_->get_territory_score());
                                double heuristic_value = territory_score / (board_size * board_size);

                                // 加权平均
                                final_value = (1.0 - weight) * nn_value + weight * heuristic_value;
                            }

                            leaf->backpropagate(final_value);
                            // --- 修改结束 ---
                    }
                }

                // === 2. 计算用于训练的策略向量 ===
                std::vector<float> action_probs(action_size, 0.0f);
                if (!root->children_.empty()) {
                    float sum_visits = 0;
                    for (const auto& child : root->children_) {
                        if (child) {
                           sum_visits += child->visit_count_;
                        }
                    }
                    // 归一化访问次数，得到策略概率
                    if (sum_visits > 0) {
                        for (const auto& child : root->children_) {
                           if (child && child->action_taken_ >= 0 && child->action_taken_ < action_size) {
                               action_probs[child->action_taken_] = static_cast<float>(child->visit_count_) / sum_visits;
                           }
                        }
                    }
                }

                 // ==================== 验证日志 开始 ====================
                                /*if (game.get_move_number() < 5) {
                                    float policy_sum = std::accumulate(action_probs.begin(), action_probs.end(), 0.0f);
                                    if (policy_sum < 0.0001f) {
                                        std::lock_guard<std::mutex> lock(g_io_mutex);
                                        std::cerr << "[VERIFICATION LOG] Game " << game_idx
                                                  << ", Move " << game.get_move_number()
                                                  << ": CRITICAL WARNING! Generated action_probs is still all zeros." << std::endl;
                                    } else {
                                        std::lock_guard<std::mutex> lock(g_io_mutex);
                                        std::cout << "[VERIFICATION LOG] Game " << game_idx
                                                  << ", Move " << game.get_move_number()
                                                  << ": Policy generation is OK. Sum=" << policy_sum << std::endl;
                                    }
                                }*/
                                // ==================== 验证日志 结束 ====================

                episode_data.emplace_back(root->game_state_->get_state(), action_probs, game.get_current_player());

                // === 3. ***保留功能点：温度采样和最终动作选择*** ===
                int action = -1;
                double current_temp = (game.get_move_number() < this->temperature_decay_moves_) ? this->temperature_start_ : this->temperature_end_;

                if (current_temp > 0.01 && !root->children_.empty()) {
                    // 温度高：按访问次数的幂次方进行概率采样，以增加探索
                    std::vector<double> powered_visits;
                    std::vector<int> actions;
                    for (const auto& child : root->children_) {
                        if (child) {
                            powered_visits.push_back(std::pow(static_cast<double>(child->visit_count_), 1.0 / current_temp));
                            actions.push_back(child->action_taken_);
                        }
                    }
                    if (!actions.empty()) {
                        static thread_local std::mt19937 generator(std::random_device{}());
                        std::discrete_distribution<int> dist(powered_visits.begin(), powered_visits.end());
                        action = actions[dist(generator)];
                    }
                // in SelfPlayManager::worker_func, inside the block for choosing an action

                } else if (!root->children_.empty()) {
                    // 温度低：贪婪选择，但随机化平局处理
                    int max_visits = -1;
                    for (const auto& child : root->children_) {
                        if (child && child->visit_count_ > max_visits) {
                            max_visits = child->visit_count_;
                        }
                    }

                    std::vector<int> best_actions;
                    for (const auto& child : root->children_) {
                        if (child && child->visit_count_ == max_visits) {
                            best_actions.push_back(child->action_taken_);
                        }
                    }

                    if (!best_actions.empty()) {
                        static thread_local std::mt19937 generator(std::random_device{}());
                        std::uniform_int_distribution<size_t> dist(0, best_actions.size() - 1);
                        action = best_actions[dist(generator)];
                    }
                }

                // 如果由于某种原因没选出动作（例如MCTS后依然没有子节点），随机选择一个
                if (action == -1) {
                    auto valid_moves = game.get_valid_moves();
                    std::vector<int> valid_move_indices;
                    for (size_t i = 0; i < valid_moves.size(); ++i) {
                        if (valid_moves[i]) valid_move_indices.push_back(i);
                    }
                    if (valid_move_indices.empty()) break; // 没有合法走法了，结束游戏
                    std::uniform_int_distribution<> distrib(0, valid_move_indices.size() - 1);
                    static thread_local std::mt19937 generator(std::random_device{}());
                    action = valid_move_indices[distrib(generator)];
                }

                // === 4. 执行动作并为下一轮做准备 ===
                game.execute_move(action);

                // 游戏结束判断与数据处理
                auto [final_value, is_done] = game.get_game_ended();
                if (is_done) {
                    // ==================== 修改数据提交流程 开始 ====================
                                        TrainingDataPacket cpp_training_examples; // 使用我们定义的新类型
                                        cpp_training_examples.reserve(episode_data.size());
                                        for (const auto& example : episode_data) {
                                            double corrected_value = final_value * std::get<2>(example);
                                            cpp_training_examples.emplace_back(std::get<0>(example), std::get<1>(example), corrected_value);
                                        }

                                        // 不再获取GIL，而是直接将C++数据包推入C++中转队列
                                        data_collector_queue_.push(std::move(cpp_training_examples));

                                        // 原子地增加已完成游戏计数
                                        completed_games_count_++;

                                        // ==================== 修改数据提交流程 结束 ====================
                                        break;
                }
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cerr << "[C++ Worker " << worker_id << ", Game " << game_idx << "] Exception: " << e.what() << std::endl;
        } catch (...) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cerr << "[C++ Worker " << worker_id << ", Game " << game_idx << "] Unknown exception occurred!" << std::endl;
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
EvaluationManager::EvaluationManager(
    std::shared_ptr<InferenceEngine> engine1,
    std::shared_ptr<InferenceEngine> engine2,
    py::dict args,
    int mode)
    : engine1_(engine1), engine2_(engine2), evaluation_mode_(mode)
{
    num_total_games_ = args["num_eval_games"].cast<int>();
    num_workers_ = args["num_cpu_threads"].cast<int>();
    num_simulations_ = args["num_eval_simulations"].cast<int>();

    // 直接从 args 读取，因为我们已在 Python 端确保了它们的传递
    board_size_ = args["board_size"].cast<int>();
    num_rounds_ = args["num_rounds"].cast<int>();
    num_channels_ = args["num_channels"].cast<int>();
    c_puct_ = args["C"].cast<double>();

    // vvvvvvvv 这是最关键的新增行 vvvvvvvv
    history_steps_ = args["history_steps"].cast<int>();
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    // 初始化计分板
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

        try {
                    Gomoku game(this->board_size_, this->num_rounds_, this->history_steps_);
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

                                auto initial_state_ptr = std::make_shared<const Gomoku>(game);
                                auto root = std::make_unique<Node>(initial_state_ptr, nullptr, -1, 1.0f);

                                // MCTS 搜索过程 (与自对弈类似，但使用C++推理引擎)
                                std::vector<Node*> leaves;
                                leaves.reserve(num_simulations_);
                                for (int i = 0; i < num_simulations_; ++i) {
                                    Node* node = root.get();
                                    node->virtual_loss_count_++;
                                    while (node->is_fully_expanded()) {
                                        node = node->select_child(this->c_puct_);
                                        node->virtual_loss_count_++;
                                    }
                                    auto [end_value, is_terminal] = node->game_state_->get_game_ended();
                                    if (is_terminal) {
                                        Node* temp_node = node;
                                                while(temp_node != nullptr) {
                                                    temp_node->virtual_loss_count_--;
                                                    temp_node = temp_node->parent_;
                                                }
                                        node->backpropagate(end_value * node->game_state_->get_current_player());
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
                                        state_batch.push_back(leaf->game_state_->get_state());
                                    }

                                    // 使用当前玩家对应的C++推理引擎
                                    auto [policy_batch, value_batch] = current_engine->infer(state_batch, this->board_size_, this->num_channels_);

                                    for (size_t i = 0; i < leaves.size(); ++i) {
                                            Node* leaf = leaves[i];
                                            const auto& current_policy = policy_batch[i];

                                            // 使用新的扩展逻辑，替换旧的 expand 调用
                                            const auto& parent_state = *leaf->game_state_;
                                            const auto valid_moves = parent_state.get_valid_moves();
                                            leaf->children_.reserve(valid_moves.size());
                                            for (size_t action = 0; action < current_policy.size(); ++action) {
                                                if (current_policy[action] > 0.0f && valid_moves[action]) {
                                                    auto next_game_state = std::make_shared<Gomoku>(parent_state);
                                                    next_game_state->execute_move(action);
                                                    leaf->children_.push_back(std::make_unique<Node>(
                                                        next_game_state, leaf, action, current_policy[action]
                                                    ));
                                                }
                                            }
                                            leaf->backpropagate(static_cast<double>(value_batch[i]));
                                        }
                                }

                                // in EvaluationManager::worker_func

                                // --- 修改开始: 选择动作（随机化平局处理和回退逻辑）---
                                int action = -1;
                                if (!root->children_.empty()) {
                                    int max_visits = -1;
                                    for (const auto& child : root->children_) {
                                        if (child && child->visit_count_ > max_visits) {
                                            max_visits = child->visit_count_;
                                        }
                                    }

                                    std::vector<int> best_actions;
                                    for (const auto& child : root->children_) {
                                        if (child && child->visit_count_ == max_visits) {
                                            best_actions.push_back(child->action_taken_);
                                        }
                                    }

                                    if (!best_actions.empty()) {
                                        // 从所有最佳动作中随机选择一个
                                        static thread_local std::mt19937 generator(std::random_device{}());
                                        std::uniform_int_distribution<size_t> dist(0, best_actions.size() - 1);
                                        action = best_actions[dist(generator)];
                                    }
                                }

                                // 如果MCTS后仍然没有选出动作（例如根节点无法扩展），则随机选择一个合法的
                                if (action == -1) {
                                    auto valid_moves = game.get_valid_moves();
                                    std::vector<int> valid_move_indices;
                                    for (size_t i = 0; i < valid_moves.size(); ++i) {
                                        if (valid_moves[i]) {
                                            valid_move_indices.push_back(i);
                                        }
                                    }
                                    if (!valid_move_indices.empty()) {
                                        static thread_local std::mt19937 generator(std::random_device{}());
                                        std::uniform_int_distribution<size_t> distrib(0, valid_move_indices.size() - 1);
                                        action = valid_move_indices[distrib(generator)];
                                    }
                                }
                                // --- 修改结束 ---


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
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(g_io_mutex);
                    std::cerr << "[C++ Eval Worker " << worker_id << ", Game " << game_idx
                              << "] Exception: " << e.what() << std::endl;
                } catch (...) {
                    std::lock_guard<std::mutex> lock(g_io_mutex);
                    std::cerr << "[C++ Eval Worker " << worker_id << ", Game " << game_idx
                              << "] Unknown exception occurred!" << std::endl;
                }


    }
}

// file: cpp_src/SelfPlayManager.cpp (替换这个函数)

int find_best_action_for_state(
    py::list py_board_pieces,
    py::list py_board_territory,
    int current_player,
    int current_move_number,
    const std::string& model_path,
    bool use_gpu,
    py::dict args)
{
    py::gil_scoped_release release;

    auto engine = get_cached_engine(model_path, use_gpu);

    double c_puct = args["C"].cast<double>();

    int board_size = args["board_size"].cast<int>();
    int max_total_moves = args.contains("max_total_moves") ? args["max_total_moves"].cast<int>() : 50;
    int history_steps = args.contains("history_steps") ? args["history_steps"].cast<int>() : 0;

    // 将Python list转换为位棋盘
    uint64_t black_s[2] = {0,0}, white_s[2] = {0,0}, black_t[2] = {0,0}, white_t[2] = {0,0};
    for (int r = 0; r < board_size; ++r) {
        py::list row = py_board_pieces[r].cast<py::list>();
        for (int c = 0; c < board_size; ++c) {
            int piece = row[c].cast<int>();
            if (piece == 0) continue;
            int pos = r * board_size + c;
            uint64_t mask = 1ULL << (pos % 64);
            if (piece == 1) black_s[pos / 64] |= mask;
            else white_s[pos / 64] |= mask;
        }
    }
    for (int r = 0; r < board_size; ++r) {
        py::list row = py_board_territory[r].cast<py::list>();
        for (int c = 0; c < board_size; ++c) {
            int territory = row[c].cast<int>();
            if (territory == 0) continue;
            int pos = r * board_size + c;
            uint64_t mask = 1ULL << (pos % 64);
            if (territory == 1) black_t[pos / 64] |= mask;
            else white_t[pos / 64] |= mask;
        }
    }

    Gomoku game(
        board_size, max_total_moves, current_player, current_move_number,
        black_s, white_s, black_t, white_t, history_steps
    );

    int num_simulations = args["num_searches"].cast<int>();
    auto initial_state_ptr = std::make_shared<const Gomoku>(game);
    auto root = std::make_unique<Node>(initial_state_ptr, nullptr, -1, 1.0f);

    // ====================== MCTS 搜索逻辑 (唯一正确的版本) ======================
    std::vector<Node*> leaves_batch;
    leaves_batch.reserve(num_simulations);
    for (int i = 0; i < num_simulations; ++i) {
        Node* node = root.get();
        node->virtual_loss_count_++;
        while (node->is_fully_expanded()) {
            node = node->select_child(c_puct);
            node->virtual_loss_count_++;
        }
        auto [end_value, is_terminal] = node->game_state_->get_game_ended();
        if (is_terminal) {
            Node* temp_node = node;
                while(temp_node != nullptr) {
                    temp_node->virtual_loss_count_--;
                    temp_node = temp_node->parent_;
                }
            node->backpropagate(end_value * node->game_state_->get_current_player());
            continue;
        }
        leaves_batch.push_back(node);
    }

    if (!leaves_batch.empty()) {
        std::sort(leaves_batch.begin(), leaves_batch.end());
        leaves_batch.erase(std::unique(leaves_batch.begin(), leaves_batch.end()), leaves_batch.end());

        std::vector<std::vector<float>> state_batch;
        state_batch.reserve(leaves_batch.size());
        for (const auto* leaf : leaves_batch) {
            state_batch.push_back(leaf->game_state_->get_state());
        }

        int num_channels = (history_steps + 1) * 4 + 4;
        auto [policy_batch, value_batch] = engine->infer(state_batch, board_size, num_channels);

        for (size_t i = 0; i < leaves_batch.size(); ++i) {
            Node* leaf = leaves_batch[i];
            const auto& current_policy = policy_batch[i];
            const auto& parent_state = *leaf->game_state_;
            const auto valid_moves = parent_state.get_valid_moves();
            leaf->children_.reserve(valid_moves.size());
            for (size_t action_idx = 0; action_idx < current_policy.size(); ++action_idx) {
                if (current_policy[action_idx] > 0.0f && valid_moves[action_idx]) {
                    auto next_game_state = std::make_shared<Gomoku>(parent_state);
                    next_game_state->execute_move(action_idx);
                    leaf->children_.push_back(std::make_unique<Node>(
                        next_game_state, leaf, action_idx, current_policy[action_idx]
                    ));
                }
            }
            leaf->backpropagate(static_cast<double>(value_batch[i]));
        }
    }
    // ====================== MCTS 逻辑结束 ======================

    // 选择最佳动作的逻辑
    int action = -1;
    if (!root->children_.empty()) {
        int max_visits = -1;
        for (const auto& child : root->children_) {
            if (child && child->visit_count_ > max_visits) {
                max_visits = child->visit_count_;
            }
        }
        std::vector<int> best_actions;
        for (const auto& child : root->children_) {
            if (child && child->visit_count_ == max_visits) {
                best_actions.push_back(child->action_taken_);
            }
        }
        if (!best_actions.empty()) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<size_t> dist(0, best_actions.size() - 1);
            action = best_actions[dist(gen)];
        }
    }

    if (action == -1) {
        auto valid_moves = game.get_valid_moves();
        std::vector<int> valid_move_indices;
        for (size_t i = 0; i < valid_moves.size(); ++i) {
            if (valid_moves[i]) valid_move_indices.push_back(i);
        }
        if (!valid_move_indices.empty()) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distrib(0, valid_move_indices.size() - 1);
            action = valid_move_indices[distrib(gen)];
        }
    }
    return action;
}