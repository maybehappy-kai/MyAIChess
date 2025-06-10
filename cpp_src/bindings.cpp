// file: cpp_src/bindings.cpp (Final Version)
#include <pybind11/pybind11.h>
#include "SelfPlayManager.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_mcts_engine, m) {
    m.doc() = "High-performance C++ engine for MCTS self-play and evaluation";

    // 绑定自对弈函数
    m.def("run_parallel_self_play", &run_parallel_self_play,
          "Starts and manages parallel self-play sessions.",
          py::arg("job_queue"),
          py::arg("result_queue"),
          py::arg("final_data_queue"),
          py::arg("args")
    );

    // 绑定评估函数
    m.def("play_game_for_eval", &play_game_for_eval,
          "Plays one game between two models for evaluation.",
          py::arg("model1"),
          py::arg("model2"),
          py::arg("args")
    );
}