// file: cpp_src/SelfPlayManager.cpp (完全修正版)
#include <pybind11/stl.h>
#include <memory>
#include <mutex>
#include <thread>
#include "SelfPlayManager.h"
#include "Gomoku.h"
#include "Node.h"
#include "SafeQueue.h"
#include "Arena.h"

struct MCTS_Config; // 先声明MCTS_Config结构体
enum class MCTS_MODE;  // 再声明MCTS_MODE枚举
std::pair<int, std::vector<float>> find_best_action_by_mcts(const Gomoku&, const std::deque<BitboardState>&, InferenceEngine&, const MCTS_Config&, MCTS_MODE);

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
std::shared_ptr<InferenceEngine> get_cached_engine(const std::string &model_path, bool use_gpu)
{
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (g_engines.find(model_path) == g_engines.end())
    {
        std::cout << "[C++ Engine Cache] Caching new model for interactive play: " << model_path << std::endl;
        g_engines[model_path] = std::make_shared<InferenceEngine>(model_path, use_gpu);
    }
    return g_engines[model_path];
}
// ==============================================================================

Gomoku reconstruct_game_state(const Node *target_node, const Gomoku &root_state)
{
    if (target_node->parent_ == nullptr)
    {
        // 如果目标节点是根节点，直接返回根状态的拷贝
        return root_state;
    }

    // 从目标节点向上遍历到根节点，收集路径上的所有动作
    std::vector<int> actions;
    const Node *current_node = target_node;
    while (current_node->parent_ != nullptr)
    {
        actions.push_back(current_node->action_taken_);
        current_node = current_node->parent_;
    }

    // 因为是从下往上收集的，所以需要反转得到正确的执行顺序
    std::reverse(actions.begin(), actions.end());

    // 从根状态开始，按顺序执行所有动作，以重构出目标节点的状态
    Gomoku reconstructed_state = root_state;
    for (int action : actions)
    {
        reconstructed_state.execute_move(action);
    }
    return reconstructed_state;
}

// ==================== 新增：历史收集辅助函数 ====================
std::deque<BitboardState> gather_history_for_leaf(
    const Node *leaf_node,
    const Gomoku &root_state,
    const std::deque<BitboardState> &root_history,
    int history_steps)
{
    // 如果不需要历史，直接返回空
    if (history_steps == 0)
    {
        return {};
    }

    // 1. 从叶子节点向上走到根节点，收集路径上的所有动作
    std::vector<int> actions;
    const Node *current_node = leaf_node;
    while (current_node != nullptr && current_node->parent_ != nullptr)
    {
        actions.push_back(current_node->action_taken_);
        current_node = current_node->parent_;
    }
    std::reverse(actions.begin(), actions.end());

    std::deque<BitboardState> full_history;
    Gomoku temp_state = root_state; // 使用根状态的轻量级拷贝

    // 2. 从根状态开始，重演所有动作，并记录路径上的每一步棋盘快照
    for (int action : actions)
    {
        full_history.push_front(temp_state.get_bitboard_state());
        temp_state.execute_move(action);
    }

    // 3. 将MCTS搜索开始前的历史（根历史）附加到路径历史之后
    full_history.insert(full_history.end(), root_history.begin(), root_history.end());

    // 4. 裁剪历史记录，确保其长度不超过 history_steps 的要求
    while (full_history.size() > static_cast<size_t>(history_steps))
    {
        full_history.pop_back();
    }

    return full_history;
}
// =============================================================

std::mutex g_io_mutex;
std::atomic<long long> g_request_id_counter(0);
std::atomic<long long> g_arena_full_count(0);

// ====================== 【新增】狄利克雷噪声辅助函数 ======================
// 在文件靠前的位置 (例如，在 g_io_mutex 定义后) 添加这个函数
void add_dirichlet_noise(
    std::vector<float> &policy,
    const std::vector<bool> &valid_moves,
    double alpha,
    double epsilon)
{
    // 使用 thread_local 确保每个工作线程拥有自己独立的、线程安全的随机数生成器
    static thread_local std::mt19937 generator(std::random_device{}());

    // 统计有多少个合法走法
    int num_valid_moves = 0;
    for (bool valid : valid_moves)
    {
        if (valid)
        {
            num_valid_moves++;
        }
    }
    // 如果合法走法少于2个，添加噪声没有意义
    if (num_valid_moves <= 1)
        return;

    // 为每个合法走法从伽马分布中采样一个值
    std::vector<float> noise_values;
    noise_values.reserve(num_valid_moves);
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float sum_of_noise = 0.0f;

    for (size_t i = 0; i < valid_moves.size(); ++i)
    {
        if (valid_moves[i])
        {
            float noise = gamma(generator);
            noise_values.push_back(noise);
            sum_of_noise += noise;
        }
    }

    // 归一化伽马分布的采样值，得到最终的狄利克雷噪声
    if (sum_of_noise > 0.0f)
    {
        for (float &n : noise_values)
        {
            n /= sum_of_noise;
        }
    }

    // 将噪声混合进原始策略中
    int noise_idx = 0;
    for (size_t i = 0; i < policy.size(); ++i)
    {
        if (valid_moves[i])
        {
            policy[i] = (1.0f - epsilon) * policy[i] + epsilon * noise_values[noise_idx];
            noise_idx++;
        }
    }
}
// ===========================================================================

// ====================== 新增：活二威胁检测启发函数 ======================
void apply_threat_detection_bias(
    std::vector<float> &policy,
    const Gomoku &game_state,
    float bonus_strength)
{
    const int board_size = game_state.get_board_size();
    const int player = game_state.get_current_player();
    auto valid_moves = game_state.get_valid_moves();

    std::vector<int> threat_moves;

    for (int action = 0; action < board_size * board_size; ++action)
    {
        if (!valid_moves[action])
            continue;

        Gomoku next_state = game_state;
        next_state.execute_move(action);

        // 在新状态下，检查对手是否已经输了 (因为我们的落子可能直接形成三连)
        // 这个检查很简单，如果对手没有合法落子点，说明我们赢了
        if (std::none_of(next_state.get_valid_moves().begin(), next_state.get_valid_moves().end(), [](bool v)
                         { return v; }))
        {
            threat_moves.push_back(action);
            continue;
        }
    }

    if (!threat_moves.empty())
    {
        float total_policy = 0.0f;
        // 给所有识别出的威胁点加上巨大的奖励
        for (int move : threat_moves)
        {
            policy[move] += bonus_strength;
        }
        // 重新归一化
        for (float p : policy)
        {
            if (p > 0)
                total_policy += p;
        }
        if (total_policy > 0)
        {
            for (float &p : policy)
            {
                if (p > 0)
                    p /= total_policy;
            }
        }
    }
}
// =======================================================================

// ==================== 新增：统一的MCTS核心函数 ====================

// 定义一个枚举来区分不同的运行模式
enum class MCTS_MODE
{
    SELF_PLAY, // 用于自对弈，启用噪声和温度采样
    EVALUATION // 用于评估或人机对战，确定性的贪婪选择
};

// file: cpp_src/SelfPlayManager.cpp
// 请用这个【完整、干净】的函数替换掉你文件中的旧函数

std::pair<int, std::vector<float>> find_best_action_by_mcts(
    const Gomoku &root_state,
    const std::deque<BitboardState> &history,
    InferenceEngine &engine,
    const MCTS_Config &config,
    MCTS_MODE mode)
{
    // 竞技场和根节点初始化 (保持不变)
    auto root = std::make_unique<Node>(nullptr, -1, 1.0f);
    Arena arena(256 * 1024 * 1024);

    // ================== MCTS 主循环 (修正循环结构) ==================
    for (int i = 0; i < config.num_simulations; i += config.mcts_batch_size)
    {
        std::vector<Node *> leaves_batch;
        leaves_batch.reserve(config.mcts_batch_size);

        // 1a. 收集叶子节点 (选择阶段)
        for (int j = 0; j < config.mcts_batch_size && i + j < config.num_simulations; ++j)
        {
            Node *node = root.get();
            node->virtual_loss_count_++; // 添加虚拟损失
            while (node->is_fully_expanded())
            {
                node = node->select_child(config.c_puct);
                node->virtual_loss_count_++;
            }

            // 检查叶子节点是否是终局状态
            Gomoku current_node_state = reconstruct_game_state(node, root_state);
            auto [end_value, is_terminal] = current_node_state.get_game_ended();

            if (is_terminal)
            {
                // 如果是终局，直接反向传播结果，无需神经网络
                Node *temp_node = node;
                while (temp_node != nullptr)
                {
                    temp_node->virtual_loss_count_--; // 偿还虚拟损失
                    temp_node = temp_node->parent_;
                }
                double value_to_propagate = end_value * current_node_state.get_current_player();
                node->backpropagate(value_to_propagate);
            }
            else
            {
                // 如果不是终局，则加入批处理队列等待扩展
                leaves_batch.push_back(node);
            }
        }

        if (leaves_batch.empty())
            continue;

        // 1b. 准备输入并进行神经网络推理 (保持不变)
        std::vector<std::vector<float>> state_batch;
        state_batch.reserve(leaves_batch.size());
        for (const auto *leaf : leaves_batch)
        {
            Gomoku leaf_state = reconstruct_game_state(leaf, root_state);
            std::deque<BitboardState> leaf_history = gather_history_for_leaf(
                leaf, root_state, history, config.history_steps);
            state_batch.push_back(leaf_state.get_state(leaf_history));
        }
        auto [policy_batch, value_batch] = engine.infer(state_batch, root_state.get_board_size(), config.num_channels);

        // ================== 1c. 扩展和反向传播 (核心修正区域) ==================
        for (size_t k = 0; k < leaves_batch.size(); ++k)
        {
            Node *leaf = leaves_batch[k];
            auto current_policy = policy_batch[k];
            Gomoku leaf_state_for_expansion = reconstruct_game_state(leaf, root_state);

            // 【修正点 1】: 将特殊逻辑(噪声/启发)与通用逻辑(扩展/回传)分离
            // 特殊逻辑只对根节点生效
            if (mode == MCTS_MODE::SELF_PLAY && leaf->parent_ == nullptr)
            {
                if (config.enable_opening_bias && leaf_state_for_expansion.get_move_number() < 50)
                {
                    // ... (开局偏置代码保持不变) ...
                    const int board_size = leaf_state_for_expansion.get_board_size();
                    float bias_strength = config.opening_bias_strength;
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
                    for (size_t pol_i = 0; pol_i < current_policy.size(); ++pol_i) {
                        current_policy[pol_i] += bias_strength * opening_bias[pol_i];
                        policy_sum += current_policy[pol_i];
                    }
                    if (policy_sum > 0.0f) {
                        for (size_t pol_i = 0; pol_i < current_policy.size(); ++pol_i) {
                            current_policy[pol_i] /= policy_sum;
                        }
                    }
                }
                if (config.enable_threat_detection)
                {
                    apply_threat_detection_bias(current_policy, leaf_state_for_expansion, config.threat_detection_bonus);
                }
                add_dirichlet_noise(current_policy, leaf_state_for_expansion.get_valid_moves(),
                                    config.dirichlet_alpha, config.dirichlet_epsilon);
            }

            // 【修正点 2】: 扩展逻辑对所有叶子节点生效
            const auto valid_moves = leaf_state_for_expansion.get_valid_moves();
            for (size_t action = 0; action < current_policy.size(); ++action)
            {
                if (current_policy[action] > 0.0f && valid_moves[action])
                {
                    try
                    {
                        Node *child_node = new (arena.allocate<Node>()) Node(leaf, action, current_policy[action]);
                        leaf->children_.push_back(child_node);
                    }
                    catch (const std::bad_alloc &)
                    {
                        g_arena_full_count++;
                        break;
                    }
                }
            }

            // 【修正点 3】: 价值计算与反向传播逻辑对所有叶子节点生效
            double nn_value = static_cast<double>(value_batch[k]);
            double final_value = nn_value;
            if (config.enable_territory_heuristic)
            {
                double territory_score = static_cast<double>(leaf_state_for_expansion.get_territory_score());
                double heuristic_value = territory_score / (root_state.get_board_size() * root_state.get_board_size());
                final_value = (1.0 - config.territory_heuristic_weight) * nn_value + config.territory_heuristic_weight * heuristic_value;
            }

            // 注意：反向传播时，要偿还之前加上的虚拟损失
            /*Node *temp_node = leaf;
            while (temp_node != nullptr) {
                temp_node->virtual_loss_count_--;
                temp_node = temp_node->parent_;
            }*/
            leaf->backpropagate(final_value);
        }
    } // ================== MCTS 主循环结束 ==================

    // ================== 【修正点 4】: 在所有模拟完成后，再计算最终策略并返回 ==================
    int action = -1;
    std::vector<float> action_probs(root_state.get_board_size() * root_state.get_board_size(), 0.0f);

    if (!root->children_.empty())
    {
        float sum_visits = 0;
        for (const auto &child : root->children_)
        {
            if (child)
                sum_visits += child->visit_count_;
        }
        if (sum_visits > 0)
        {
            for (const auto &child : root->children_)
            {
                if (child)
                    action_probs[child->action_taken_] = static_cast<float>(child->visit_count_) / sum_visits;
            }
        }
    }

    // 根据模式选择最终动作 (温度采样或贪心选择)
    double temperature = 0.0;
    if (mode == MCTS_MODE::SELF_PLAY)
    {
        temperature = (root_state.get_move_number() < config.temperature_decay_moves)
                          ? config.temperature_start
                          : config.temperature_end;
    }

    if (temperature > 0.01 && !root->children_.empty())
    {
        // 温度采样
        std::vector<double> powered_visits;
        std::vector<int> actions;
        for (const auto &child : root->children_)
        {
            if (child)
            {
                powered_visits.push_back(std::pow(static_cast<double>(child->visit_count_), 1.0 / temperature));
                actions.push_back(child->action_taken_);
            }
        }
        if (!actions.empty())
        {
            static thread_local std::mt19937 generator(std::random_device{}());
            std::discrete_distribution<int> dist(powered_visits.begin(), powered_visits.end());
            action = actions[dist(generator)];
        }
    }
    else if (!root->children_.empty())
    {
        // 贪婪选择
        int max_visits = -1;
        std::vector<int> best_actions;
        for (const auto &child : root->children_)
        {
            if (child && child->visit_count_ > max_visits)
            {
                max_visits = child->visit_count_;
                best_actions.clear();
                best_actions.push_back(child->action_taken_);
            } else if (child && child->visit_count_ == max_visits) {
                best_actions.push_back(child->action_taken_);
            }
        }
        if (!best_actions.empty())
        {
            static thread_local std::mt19937 generator(std::random_device{}());
            std::uniform_int_distribution<size_t> dist(0, best_actions.size() - 1);
            action = best_actions[dist(generator)];
        }
    }

    // 如果没选出动作 (例如，根节点是终局)，随机选择一个合法动作
    if (action == -1)
    {
        auto valid_moves = root_state.get_valid_moves();
        std::vector<int> valid_move_indices;
        for (size_t i = 0; i < valid_moves.size(); ++i)
        {
            if (valid_moves[i])
                valid_move_indices.push_back(i);
        }
        if (!valid_move_indices.empty())
        {
            std::uniform_int_distribution<> distrib(0, valid_move_indices.size() - 1);
            static thread_local std::mt19937 generator(std::random_device{}());
            action = valid_move_indices[distrib(generator)];
        }
    }

    return {action, action_probs};
}

// =============================================================

// Python调用的顶层函数
void run_parallel_self_play(const std::string &model_path, bool use_gpu, py::object final_data_queue, py::dict args)
{
    auto engine = std::make_shared<InferenceEngine>(model_path, use_gpu);
    auto manager = std::make_shared<SelfPlayManager>(engine, final_data_queue, args);

    std::thread cpp_manager_thread([manager]()
                                   { manager->run(); });

    {
        py::gil_scoped_release release;
        if (cpp_manager_thread.joinable())
        {
            cpp_manager_thread.join();
        }
    }
}

SelfPlayManager::SelfPlayManager(std::shared_ptr<InferenceEngine> engine, py::object final_data_queue, py::dict args)
    : engine_(engine), final_data_queue_(final_data_queue)
{
    this->num_total_games_ = args["num_selfPlay_episodes"].cast<int>();
    this->num_workers_ = args["num_cpu_threads"].cast<int>();
    this->board_size_ = args["board_size"].cast<int>();
    this->num_rounds_ = args["num_rounds"].cast<int>();

    // 直接填充MCTS配置结构体
    this->mcts_config_.num_simulations = args["num_searches"].cast<int>();
    this->mcts_config_.mcts_batch_size = args["mcts_batch_size"].cast<int>();
    this->mcts_config_.c_puct = args["C"].cast<double>();
    this->mcts_config_.history_steps = args["history_steps"].cast<int>();
    this->mcts_config_.num_channels = args["num_channels"].cast<int>();
    this->mcts_config_.enable_opening_bias = args["enable_opening_bias"].cast<bool>();
    this->mcts_config_.opening_bias_strength = args["opening_bias_strength"].cast<float>();
    this->mcts_config_.enable_threat_detection = args["enable_threat_detection"].cast<bool>();
    this->mcts_config_.threat_detection_bonus = args["threat_detection_bonus"].cast<float>();
    this->mcts_config_.enable_territory_heuristic = args["enable_territory_heuristic"].cast<bool>();
    this->mcts_config_.territory_heuristic_weight = args["territory_heuristic_weight"].cast<double>();
    this->mcts_config_.dirichlet_alpha = args["dirichlet_alpha"].cast<double>();
    this->mcts_config_.dirichlet_epsilon = args["dirichlet_epsilon"].cast<double>();
    this->mcts_config_.temperature_start = args["temperature_start"].cast<double>();
    this->mcts_config_.temperature_end = args["temperature_end"].cast<double>();
    this->mcts_config_.temperature_decay_moves = args["temperature_decay_moves"].cast<int>();
}
// file: cpp_src/SelfPlayManager.cpp
// 找到 SelfPlayManager::collector_func 函数，并用下面的完整版本替换它

void SelfPlayManager::collector_func()
{
    // 主循环：在游戏还在进行时，持续搬运数据
    while (completed_games_count_ < num_total_games_)
    {
        TrainingDataPacket packet;
        if (data_collector_queue_.try_pop(packet))
        {
            // 成功从中转站取出数据包
            // 这部分逻辑保持不变
            py::gil_scoped_acquire acquire;
            py::list training_examples_list;
            for (const auto &ex : packet)
            {
                training_examples_list.append(py::make_tuple(
                    py::cast(std::get<0>(ex)),
                    py::cast(std::get<1>(ex)),
                    py::cast(std::get<2>(ex))));
            }
            py::dict data_to_send;
            data_to_send["type"] = "data";
            data_to_send["data"] = training_examples_list;
            final_data_queue_.attr("put")(data_to_send);
        }
        else
        {
            // 中转站暂时为空，短暂休眠一下，避免空转浪费CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // ====================== 【新增的收尾工作】 ======================
    // 主循环退出后（所有游戏都已完成），为防止有数据在竞态中被遗漏，
    // 我们需要在这里彻底清空一次队列。
    TrainingDataPacket packet;
    while (data_collector_queue_.try_pop(packet))
    {
        // 这里的逻辑和主循环中完全一样，确保最后的数据也被处理
        py::gil_scoped_acquire acquire;
        py::list training_examples_list;
        for (const auto &ex : packet)
        {
            training_examples_list.append(py::make_tuple(
                py::cast(std::get<0>(ex)),
                py::cast(std::get<1>(ex)),
                py::cast(std::get<2>(ex))));
        }
        py::dict data_to_send;
        data_to_send["type"] = "data";
        data_to_send["data"] = training_examples_list;
        final_data_queue_.attr("put")(data_to_send);
    }
    // ==============================================================
}

void SelfPlayManager::run()
{
    for (int i = 0; i < this->num_total_games_; ++i)
    {
        task_queue_.push(i);
    }

    {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[C++ Manager] Starting " << this->num_workers_ << " worker threads to play " << this->num_total_games_ << " games..." << std::endl;
    }

    // ==================== 新增：启动搬运工线程 ====================
    collector_thread_ = std::thread(&SelfPlayManager::collector_func, this);

    threads_.reserve(this->num_workers_);
    for (int i = 0; i < this->num_workers_; ++i)
    {
        threads_.emplace_back(&SelfPlayManager::worker_func, this, i);
    }

    for (auto &t : threads_)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    // ==================== 新增：等待搬运工线程结束 ====================
    if (collector_thread_.joinable())
    {
        collector_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[C++ Manager] All self-play games finished." << std::endl;
        // 打印Arena耗尽的总次数
        std::cout << "[C++ Manager] Arena was exhausted " << g_arena_full_count.load() << " times." << std::endl;
        g_arena_full_count = 0; // 重置以便下次迭代
    }
}

// 文件: cpp_src/SelfPlayManager.cpp
// 这是包含了您指出的 action_size 初始化代码的最终修正版

void SelfPlayManager::worker_func(int worker_id)
{
    while (true)
    {
        int game_idx;
        if (!task_queue_.try_pop(game_idx))
        {
            break; // 任务队列空，线程退出
        }

        try
        {
            Gomoku game(this->board_size_, this->num_rounds_, this->mcts_config_.history_steps);
            std::deque<BitboardState> game_history;
            std::vector<std::tuple<std::vector<float>, std::vector<float>, int>> episode_data;

            // 【已加回】根据您的反馈，保留 action_size 的初始化
            const int action_size = game.get_board_size() * game.get_board_size();

            while (true)
            {
                // 1. 【先判断】游戏是否结束
                auto [final_value, is_done] = game.get_game_ended();
                if (is_done)
                {
                    // 如果结束，整理并提交本局所有数据，然后跳出循环
                    TrainingDataPacket cpp_training_examples;
                    cpp_training_examples.reserve(episode_data.size());
                    for (const auto &example : episode_data)
                    {
                        double corrected_value = final_value * std::get<2>(example);
                        cpp_training_examples.emplace_back(std::get<0>(example), std::get<1>(example), corrected_value);
                    }
                    data_collector_queue_.push(std::move(cpp_training_examples));
                    completed_games_count_++;
                    break; // 结束本局游戏
                }

                // 2. 【后执行】如果游戏未结束，才为当前状态寻找最佳动作
                const Gomoku root_state(game);
                auto [action, action_probs] = find_best_action_by_mcts(
                    root_state,
                    game_history,
                    *engine_,
                    this->mcts_config_,
                    MCTS_MODE::SELF_PLAY);

                // 3. 存储【有效】的训练数据
                episode_data.emplace_back(root_state.get_state(game_history), action_probs, game.get_current_player());

                // 4. 更新历史记录并执行动作
                game_history.push_front(game.get_bitboard_state());
                if (game_history.size() > static_cast<size_t>(this->mcts_config_.history_steps))
                {
                    game_history.pop_back();
                }
                game.execute_move(action);
            }
        }
        catch (const std::exception &e)
        {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cerr << "[C++ Worker " << worker_id << ", Game " << game_idx << "] Exception: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cerr << "[C++ Worker " << worker_id << ", Game " << game_idx << "] Unknown exception occurred!" << std::endl;
        }
    }
}

// ====================== 新增：高效并行评估的完整实现 ======================

// C++评估任务的顶层入口
py::dict run_parallel_evaluation(const std::string &model1_path, const std::string &model2_path, bool use_gpu, py::dict args, int mode)
{ // <-- 增加mode参数
    auto engine1 = std::make_shared<InferenceEngine>(model1_path, use_gpu);
    auto engine2 = std::make_shared<InferenceEngine>(model2_path, use_gpu);

    auto eval_manager = std::make_shared<EvaluationManager>(engine1, engine2, args, mode); // <-- 将mode传给构造函数

    std::thread cpp_eval_thread([eval_manager]()
                                { eval_manager->run(); });

    {
        py::gil_scoped_release release;
        if (cpp_eval_thread.joinable())
        {
            cpp_eval_thread.join();
        }
    }

    return eval_manager->get_results();
}

EvaluationManager::EvaluationManager(
    std::shared_ptr<InferenceEngine> engine1,
    std::shared_ptr<InferenceEngine> engine2,
    py::dict args,
    int mode)
    : engine1_(engine1), engine2_(engine2), evaluation_mode_(mode)
{
    // 非 MCTS 参数
    this->num_total_games_ = args["num_eval_games"].cast<int>();
    this->num_workers_ = args["num_cpu_threads"].cast<int>();
    this->board_size_ = args["board_size"].cast<int>();
    this->num_rounds_ = args["num_rounds"].cast<int>();

    // 直接填充 MCTS 配置结构体
    // 注意：评估使用的是 num_eval_simulations
    this->mcts_config_.num_simulations = args["num_eval_simulations"].cast<int>();
    this->mcts_config_.c_puct = args["C"].cast<double>();
    this->mcts_config_.mcts_batch_size = args["mcts_batch_size"].cast<int>();
    this->mcts_config_.history_steps = args["history_steps"].cast<int>();
    this->mcts_config_.num_channels = args["num_channels"].cast<int>();

    // 评估模式下，以下参数不起作用，但为保持结构完整性，我们仍进行初始化
    this->mcts_config_.enable_opening_bias = false;
    this->mcts_config_.opening_bias_strength = 0.0f;
    this->mcts_config_.enable_threat_detection = false;
    this->mcts_config_.threat_detection_bonus = 0.0f;
    // 从Python的args字典中动态读取领地启发参数
    this->mcts_config_.enable_territory_heuristic = args.contains("enable_territory_heuristic") ? args["enable_territory_heuristic"].cast<bool>() : false;
    this->mcts_config_.territory_heuristic_weight = args.contains("territory_heuristic_weight") ? args["territory_heuristic_weight"].cast<double>() : 0.0;
    this->mcts_config_.dirichlet_alpha = 0.0;
    this->mcts_config_.dirichlet_epsilon = 0.0;
    this->mcts_config_.temperature_start = 0.0;
    this->mcts_config_.temperature_end = 0.0;
    this->mcts_config_.temperature_decay_moves = 0;

    this->scores_[1] = 0;   // 为"模型1胜利"初始化
    this->scores_[-1] = 0;  // 为"模型2胜利"初始化
    this->scores_[0] = 0;   // 为"平局"初始化
}

py::dict EvaluationManager::get_results() const
{
    py::dict results;
    results["model1_wins"] = scores_.at(1);
    results["model2_wins"] = scores_.at(-1);
    results["draws"] = scores_.at(0);
    return results;
}

void EvaluationManager::run()
{
    for (int i = 0; i < num_total_games_; ++i)
    {
        task_queue_.push(i);
    }
    threads_.reserve(num_workers_);
    for (int i = 0; i < num_workers_; ++i)
    {
        threads_.emplace_back(&EvaluationManager::worker_func, this, i);
    }
    for (auto &t : threads_)
    {
        if (t.joinable())
            t.join();
    }
}

// 评估工作线程，核心逻辑在此

void EvaluationManager::worker_func(int worker_id)
{
    while (true)
    {
        int game_idx;
        if (!task_queue_.try_pop(game_idx))
        {
            break;
        }

        try
        {
            Gomoku game(this->board_size_, this->num_rounds_, this->mcts_config_.history_steps);
            std::deque<BitboardState> game_history; // <-- 同样需要历史记录deque

            // ... (选择p1_engine和p2_engine的逻辑保持不变) ...
            auto &p1_engine = engine1_;
            auto &p2_engine = engine2_;
            if (evaluation_mode_ == 0 && (game_idx % 2 != 0))
            { // Mode 0: 交替先后手

                bool swap_models = (game_idx % 2 != 0);

                if (swap_models)
                {

                    p1_engine = engine2_;

                    p2_engine = engine1_;
                }
            }
            else if (evaluation_mode_ == 2)
            { // Mode 2: 固定 Model 2 先手

                p1_engine = engine2_;

                p2_engine = engine1_;
            }

            std::map<int, std::shared_ptr<InferenceEngine>> models = {
                {1, p1_engine}, {-1, p2_engine}};

            while (true)
            {
                const Gomoku root_state(game);
                auto &current_engine = models.at(game.get_current_player());

                // 调用统一的MCTS核心函数，模式为EVALUATION
                auto [action, policy] = find_best_action_by_mcts(
                    root_state,
                    game_history,
                    *current_engine,
                    this->mcts_config_,
                    MCTS_MODE::EVALUATION // 指定为评估模式
                );
                // 在评估中，我们不关心返回的policy，所以可以忽略

                if (action == -1)
                    break;

                // 更新历史并执行动作
                game_history.push_front(game.get_bitboard_state());
                if (game_history.size() > static_cast<size_t>(this->mcts_config_.history_steps))
                {
                    game_history.pop_back();
                }
                game.execute_move(action);

                {
                        std::lock_guard<std::mutex> lock(g_io_mutex); // 使用全局锁确保打印不混乱
                        std::cout << "\n=======================================================\n";
                        std::cout << "[Eval Game " << game_idx << ", Worker " << worker_id << "] Move #" << game.get_move_number() << "\n";
                        std::cout << "Player " << game.get_current_player() * -1 << " (Engine: " << ((current_engine == engine1_) ? "Model1-Old" : "Model2-New") << ") chose action: " << action << "\n";
                        game.print_board();
                        std::cout << "=======================================================\n";
                    }

                auto [final_value, is_done] = game.get_game_ended();
                if (is_done)
                {
                    int winner_code = 0; // 0 for draw
                    if (std::abs(final_value) > 0.01)
                    {
                        int winner_player = static_cast<int>(final_value); // 1 for P1 (black), -1 for P2 (white)

                        // 判断胜利方的引擎是哪一个
                        std::shared_ptr<InferenceEngine> winning_engine;
                        if (winner_player == 1)
                        {
                            winning_engine = p1_engine;
                        }
                        else
                        { // winner_player == -1
                            winning_engine = p2_engine;
                        }

                        // 无论谁赢，都统一检查胜利的引擎是 model1 还是 model2
                        if (winning_engine == engine1_)
                        {
                            winner_code = 1; // Model 1 (旧模型) 胜利
                        }
                        else
                        {
                            winner_code = -1; // Model 2 (新模型) 胜利
                        }
                    }

                    {
                        std::lock_guard<std::mutex> lock(results_mutex_);
                        scores_[winner_code]++;
                    }

                    break;
                }
            }
        }
        catch (const std::exception &e)
        {

            std::lock_guard<std::mutex> lock(g_io_mutex);

            std::cerr << "[C++ Eval Worker " << worker_id << ", Game " << game_idx

                      << "] Exception: " << e.what() << std::endl;
        }
        catch (...)
        {

            std::lock_guard<std::mutex> lock(g_io_mutex);

            std::cerr << "[C++ Eval Worker " << worker_id << ", Game " << game_idx

                      << "] Unknown exception occurred!" << std::endl;
        }
    }
}

// file: cpp_src/SelfPlayManager.cpp
// 用这个【最终、完全修正版】的函数，替换掉文件中旧的同名函数

int find_best_action_for_state(
    py::list py_board_pieces,
    py::list py_board_territory,
    int current_player,
    int current_move_number,
    const std::string &model_path,
    bool use_gpu,
    py::dict args)
{
    // 1. 获取模型引擎
    auto engine = get_cached_engine(model_path, use_gpu);

    // 2. 从Python传入的args字典，动态创建MCTS_Config结构体
    MCTS_Config config;
    config.num_simulations = args["num_searches"].cast<int>();
    config.c_puct = args["C"].cast<double>();
    config.mcts_batch_size = args["mcts_batch_size"].cast<int>();
    config.history_steps = args["history_steps"].cast<int>();

    // ====================== 【核心修正】 ======================
    // 修正了通道数的计算逻辑，确保与Python训练时完全一致
    int history_steps = args["history_steps"].cast<int>();
    int state_channels = (history_steps + 1) * 4; // (历史步数 + 当前状态) * 4个平面
    int meta_channels = 4; // 4个元数据平面
    config.num_channels = state_channels + meta_channels; // 总通道数，(3+1)*4 + 4 = 20
    // =========================================================

    // 在评估/对战模式下，其他启发式参数设为默认关闭状态
    config.enable_opening_bias = false;
    config.opening_bias_strength = 0.0f;
    config.enable_threat_detection = false;
    config.threat_detection_bonus = 0.0f;
    // 从Python的args字典中动态读取领地启发参数
    config.enable_territory_heuristic = args.contains("enable_territory_heuristic") ? args["enable_territory_heuristic"].cast<bool>() : false;
    config.territory_heuristic_weight = args.contains("territory_heuristic_weight") ? args["territory_heuristic_weight"].cast<double>() : 0.0;
    config.dirichlet_alpha = 0.0;
    config.dirichlet_epsilon = 0.0;
    config.temperature_start = 0.0;
    config.temperature_end = 0.0;
    config.temperature_decay_moves = 0;

    // 3. 从Python列表恢复Gomoku的根状态
    int board_size = args["board_size"].cast<int>();
    int max_total_moves = args["max_total_moves"].cast<int>();
    uint64_t black_s[2] = {0, 0}, white_s[2] = {0, 0}, black_t[2] = {0, 0}, white_t[2] = {0, 0};
    for (int r = 0; r < board_size; ++r)
    {
        py::list row_p = py_board_pieces[r].cast<py::list>();
        py::list row_t = py_board_territory[r].cast<py::list>();
        for (int c = 0; c < board_size; ++c)
        {
            int piece = row_p[c].cast<int>();
            if (piece != 0) {
                int pos = r * board_size + c;
                uint64_t mask = 1ULL << (pos % 64);
                if (piece == 1) black_s[pos / 64] |= mask; else white_s[pos / 64] |= mask;
            }
            int territory = row_t[c].cast<int>();
            if (territory != 0) {
                int pos = r * board_size + c;
                uint64_t mask = 1ULL << (pos % 64);
                if (territory == 1) black_t[pos / 64] |= mask; else white_t[pos / 64] |= mask;
            }
        }
    }

    // 【修正】Gomoku的构造函数需要的是 num_rounds，而不是 max_total_moves
    int num_rounds = max_total_moves / 2;
    Gomoku root_state(
        board_size, num_rounds,
        current_player, current_move_number,
        black_s, white_s, black_t, white_t, config.history_steps
    );

    // 为人机对战创建一个空的history deque
    std::deque<BitboardState> history;

    int final_action = -1;
    {
        // 仅在进行纯C++的MCTS重度计算时，才释放GIL
        py::gil_scoped_release release;

        // 调用统一的MCTS核心函数
        auto [best_action, policy] = find_best_action_by_mcts(
            root_state,
            history,
            *engine,
            config,
            MCTS_MODE::EVALUATION
        );
        final_action = best_action;
    }

    // 返回最终动作
    return final_action;
}