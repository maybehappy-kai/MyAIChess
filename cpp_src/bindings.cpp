// file: cpp_src/bindings.cpp (Final Version)
#include <pybind11/pybind11.h>
#include "SelfPlayManager.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_mcts_engine, m) {
    m.doc() = "High-performance C++ engine for MCTS self-play and evaluation";

    // --- vvv 这里是需要修改的地方 vvv ---

        // 绑定新的自对弈函数接口
        m.def("run_parallel_self_play", &run_parallel_self_play,
              "Starts and manages parallel self-play sessions using C++ inference.",
              py::arg("model_path"),       // 1. 修改：不再是 job_queue
              py::arg("use_gpu"),          // 2. 新增：告诉C++是否使用GPU
              py::arg("final_data_queue"), // 3. 保留：用于接收最终结果
              py::arg("args")              // 4. 保留：传递超参数
        );

        // --- ^^^ 修改结束 ^^^ ---

    // 绑定评估函数
    m.def("play_game_for_eval", &play_game_for_eval,
          "Plays one game between two models for evaluation.",
          py::arg("model1"),
          py::arg("model2"),
          py::arg("args")
    );
}