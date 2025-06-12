// file: cpp_src/SelfPlayManager.h
#pragma once
#include <pybind11/pybind11.h>
#include <thread>
#include <vector>
#include <mutex>
#include <map>
#include "SafeQueue.h" // 引入线程安全队列
#include "InferenceEngine.h" // <-- 1. 新增: 包含我们新创建的引擎头文件
#include <memory>            // <-- 2. 新增: 包含智能指针的头文件

extern std::mutex g_io_mutex;

namespace py = pybind11;

class SelfPlayManager {
public:
    // <-- 3. 修改: 构造函数签名，不再接收Python队列，而是接收C++引擎
    SelfPlayManager(std::shared_ptr<InferenceEngine> engine, py::object final_data_queue, py::dict args);
    void run();
private:
    void worker_func(int worker_id);

    // <-- 4. 移除: 不再需要的Python推理队列
        // py::object job_queue_;
        // py::object result_queue_;
        // py::object queue_empty_exc_;

        py::object final_data_queue_;
        py::dict args_;

        std::shared_ptr<InferenceEngine> engine_; // <-- 5. 新增: C++推理引擎的智能指针成员


    // 线程池与任务队列
    std::vector<std::thread> threads_;
    int num_workers_; // 现在是线程池大小
    int num_total_games_; // 总共要玩的游戏数
    SafeQueue<int> task_queue_; // C++内部的任务队列
};

// ... 函数声明保持不变 ...

// <-- 6. 修改: 顶层函数的声明，以匹配新的调用方式
void run_parallel_self_play(const std::string& model_path, bool use_gpu, py::object final_data_queue, py::dict args);

// 新增：评估模式入口函数声明
int play_game_for_eval(py::object model1, py::object model2, py::dict args);