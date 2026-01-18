// file: cpp_src/SelfPlayManager.h (终极重构版)
#pragma once
#include <pybind11/pybind11.h>
#include <thread>
#include <vector>
#include <mutex>
#include <map>
#include "SafeQueue.h"
#include "InferenceEngine.h"
#include <memory>
#include <cstdint>

extern std::mutex g_io_mutex;

namespace py = pybind11;

// ==================== MCTS配置结构体 (保持不变) ====================
struct MCTS_Config
{
    int num_simulations;
    int mcts_batch_size;
    double c_puct;
    int history_steps;
    int num_channels;
    bool enable_opening_bias;
    float opening_bias_strength;
    bool enable_ineffective_connection_penalty;
    float ineffective_connection_penalty_factor;
    bool enable_territory_heuristic;
    double territory_heuristic_weight;
    double dirichlet_alpha;
    double dirichlet_epsilon;
    double temperature_start;
    double temperature_end;
    int temperature_decay_moves;
    bool enable_territory_penalty;
    float territory_penalty_strength;
};
// =================================================================

struct EvalInitialState {
    uint64_t black_stones[2];
    uint64_t white_stones[2];
    uint64_t black_territory[2];
    uint64_t white_territory[2];
    int current_player;
    int current_move_number;
};

using TrainingDataPacket = std::vector<std::tuple<std::vector<float>, std::vector<float>, double>>;

class SelfPlayManager
{
public:
    SelfPlayManager(std::shared_ptr<InferenceEngine> engine, py::object final_data_queue, py::dict args);
    void run();

private:
    void worker_func(int worker_id);
    void collector_func();

    py::object final_data_queue_;
    std::shared_ptr<InferenceEngine> engine_;
    std::vector<std::thread> threads_;
    SafeQueue<int> task_queue_;
    SafeQueue<TrainingDataPacket> data_collector_queue_;
    std::thread collector_thread_;
    std::atomic<int> completed_games_count_{0};

    // ====================== SelfPlayManager 核心修正区域 ======================
    // 将零散的MCTS参数聚合到 MCTS_Config 结构体中
    int num_total_games_;
    int num_workers_;
    MCTS_Config mcts_config_; // <-- 使用一个结构体替换下面的所有零散参数
    int board_size_;
    int num_rounds_;
    /*
    // 以下成员变量已被移入 MCTS_Config
    int num_simulations_;
    int mcts_batch_size_;
    int history_steps_;
    int num_channels_;
    double dirichlet_alpha_;
    double dirichlet_epsilon_;
    double temperature_start_;
    double temperature_end_;
    int temperature_decay_moves_;
    double c_puct_;
    bool   enable_opening_bias_;
    float  opening_bias_strength_;
    bool   enable_threat_detection_;
    float  threat_detection_bonus_;
    bool   enable_territory_heuristic_;
    double territory_heuristic_weight_;
    */
    // =======================================================================
};

// ====================== 评估 (Evaluation) 部分的声明 ======================
class EvaluationManager
{
public:
    EvaluationManager(
        std::shared_ptr<InferenceEngine> engine1,
        std::shared_ptr<InferenceEngine> engine2,
        py::dict args,
        int mode,
        py::list initial_states); // <--- 新增参数
    void run();
    py::dict get_results() const;

private:
    void worker_func(int worker_id);

    std::shared_ptr<InferenceEngine> engine1_;
    std::shared_ptr<InferenceEngine> engine2_;
    std::vector<std::thread> threads_;
    SafeQueue<int> task_queue_;
    std::vector<EvalInitialState> initial_states_;

    // ====================== EvaluationManager 核心修正区域 ======================
    // 同样，将MCTS参数聚合到 MCTS_Config 结构体中
    int num_total_games_;
    int num_workers_;
    int evaluation_mode_;
    MCTS_Config mcts_config_; // <-- 使用一个结构体替换下面的所有零散参数
    int board_size_;
    int num_rounds_;
    /*
    // 以下成员变量已被移入 MCTS_Config
    int num_simulations_;
    int num_channels_;
    int history_steps_;
    double c_puct_;
    */
    // =======================================================================

    std::mutex results_mutex_;
    std::map<int, int> scores_;
};

// --- 顶层函数声明 (保持不变) ---
void run_parallel_self_play(const std::string &model_path, bool use_gpu, py::object final_data_queue, py::dict args);

py::dict run_parallel_evaluation(
    const std::string &model1_path,
    const std::string &model2_path,
    bool use_gpu,
    py::dict args,
    int mode,
    py::list initial_states); // <--- 新增参数

int find_best_action_for_state(
    py::list board_pieces,
    py::list board_territory,
    int current_player,
    int current_move_number,
    const std::string &model_path,
    bool use_gpu,
    py::dict args);