// file: cpp_src/SelfPlayManager.h
#pragma once
#include <pybind11/pybind11.h>
#include <thread>
#include <vector>
#include <mutex>
#include <map>
#include "SafeQueue.h" // 引入线程安全队列

extern std::mutex g_io_mutex;

namespace py = pybind11;

class SelfPlayManager {
public:
    SelfPlayManager(py::object job_queue, py::object result_queue, py::object final_data_queue, py::dict args);
    void run();
private:
    void worker_func(int worker_id);

    // Python对象
    py::object job_queue_;
    py::object result_queue_;
    py::object final_data_queue_;
    py::dict args_;
    py::object queue_empty_exc_;

    // 线程池与任务队列
    std::vector<std::thread> threads_;
    int num_workers_; // 现在是线程池大小
    int num_total_games_; // 总共要玩的游戏数
    SafeQueue<int> task_queue_; // C++内部的任务队列
};

// ... 函数声明保持不变 ...

// 自对弈入口函数声明
void run_parallel_self_play(py::object job_queue, py::object result_queue, py::object final_data_queue, py::dict args);

// 新增：评估模式入口函数声明
int play_game_for_eval(py::object model1, py::object model2, py::dict args);