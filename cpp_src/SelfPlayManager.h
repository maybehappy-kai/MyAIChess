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
int play_game_for_eval(py::object model1, py::object model2, py::dict args);