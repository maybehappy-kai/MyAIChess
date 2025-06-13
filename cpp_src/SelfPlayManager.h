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

class SelfPlayManager {
public:
    SelfPlayManager(std::shared_ptr<InferenceEngine> engine, py::object final_data_queue, py::dict args);
    void run();

private:
    void worker_func(int worker_id);

    // Python对象
    py::object final_data_queue_;

    // C++推理引擎
    std::shared_ptr<InferenceEngine> engine_;

    // 线程池与任务管理
    std::vector<std::thread> threads_;
    SafeQueue<int> task_queue_;

    // ====================== 核心修正区域 ======================
    // 不再持有 py::dict, 而是将参数用C++原生类型存储，避免生命周期问题
    int num_total_games_;
    int num_workers_;
    int num_simulations_;
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