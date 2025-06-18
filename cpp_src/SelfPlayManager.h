// file: cpp_src/SelfPlayManager.h (完全修正版)
#pragma once
#include <pybind11/pybind11.h>
#include <thread>
#include <vector>
#include <mutex>
#include <map>
#include "SafeQueue.h"
#include "InferenceEngine.h"
#include <memory>

extern std::mutex g_io_mutex;

namespace py = pybind11;

// ==================== 新增：定义一个清晰的数据包类型 ====================
using TrainingDataPacket = std::vector<std::tuple<std::vector<float>, std::vector<float>, double>>;

class SelfPlayManager {
public:
    SelfPlayManager(std::shared_ptr<InferenceEngine> engine, py::object final_data_queue, py::dict args);
    void run();

private:
    void worker_func(int worker_id);

    // ==================== 新增：搬运工线程的声明 ====================
        void collector_func();

        py::object final_data_queue_;
        std::shared_ptr<InferenceEngine> engine_;
        std::vector<std::thread> threads_;
        SafeQueue<int> task_queue_;

        // ==================== 新增：C++数据中转站和搬运工线程 ====================
        SafeQueue<TrainingDataPacket> data_collector_queue_;
        std::thread collector_thread_;
        std::atomic<int> completed_games_count_{0}; // 用于通知搬运工何时下班

    // ====================== 核心修正区域 ======================
    // 不再持有 py::dict, 而是将参数用C++原生类型存储，避免生命周期问题
    int num_total_games_;
    int num_workers_;
    int num_simulations_;
    int mcts_batch_size_;      // MCTS推理时的批处理大小
    int board_size_;
        int num_rounds_;
        int history_steps_;
        int num_channels_;

    double dirichlet_alpha_;   // <--- 【新增此行】
        double dirichlet_epsilon_; // <--- 【新增此行】
        // 【新增】温度采样相关的成员变量
            double temperature_start_;
            double temperature_end_;
            int temperature_decay_moves_;

            double c_puct_;

        // =======================================================
            // --- 核心改动：用C++原生类型替换py::dict args_ ---
            bool   enable_opening_bias_;
            float  opening_bias_strength_;
            bool   enable_threat_detection_;
            float  threat_detection_bonus_;
            bool   enable_territory_heuristic_;
            double territory_heuristic_weight_;
            // =======================================================
    // =========================================================
};

// 函数声明
void run_parallel_self_play(const std::string& model_path, bool use_gpu, py::object final_data_queue, py::dict args);
//int play_game_for_eval(py::object model1, py::object model2, py::dict args);

// ====================== 新增评估部分的声明 ======================
class EvaluationManager {
public:
    EvaluationManager(
        std::shared_ptr<InferenceEngine> engine1,
        std::shared_ptr<InferenceEngine> engine2,
        py::dict args,
        int mode // <-- 新增mode参数
    );
    void run();
    py::dict get_results() const;

private:
    void worker_func(int worker_id);

    std::shared_ptr<InferenceEngine> engine1_;
    std::shared_ptr<InferenceEngine> engine2_;

    std::vector<std::thread> threads_;
    SafeQueue<int> task_queue_;

    int num_total_games_;
    int num_workers_;
    int num_simulations_;
    int evaluation_mode_; // <-- 新增成员变量
    int board_size_;
        int num_rounds_;
        int num_channels_;
        int history_steps_;

        double c_puct_;

    // 用于存储结果的线程安全成员
    std::mutex results_mutex_;
    std::map<int, int> scores_; // 1: model1 wins, -1: model2 wins, 0: draw
};

// 新的顶层评估函数声明
py::dict run_parallel_evaluation(
    const std::string& model1_path,
    const std::string& model2_path,
    bool use_gpu,
    py::dict args,
    int mode // <-- 新增mode参数
);
// =============================================================

// ====================== 新增单步决策函数的声明 ======================
int find_best_action_for_state(
    py::list board_pieces,
    py::list board_territory,
    int current_player,
    int current_move_number,
    const std::string& model_path,
    bool use_gpu,
    py::dict args
);
// =============================================================