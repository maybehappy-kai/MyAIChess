// file: cpp_src/bindings.cpp (修正后)
#include <pybind11/pybind11.h>
#include "SelfPlayManager.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_mcts_engine, m) {
    m.doc() = "High-performance C++ engine for MCTS self-play and evaluation";

    // 自对弈函数接口 (这个是正确的)
    m.def("run_parallel_self_play", &run_parallel_self_play,
          "Starts and manages parallel self-play sessions using C++ inference.",
          py::arg("model_path"),
          py::arg("use_gpu"),
          py::arg("final_data_queue"),
          py::arg("args")
    );

    // 【核心修正】绑定新的并行评估函数接口
    m.def("run_parallel_evaluation", &run_parallel_evaluation,
          "Plays multiple games in parallel between two models for evaluation using C++ engines.",
          py::arg("model1_path"),
          py::arg("model2_path"),
          py::arg("use_gpu"),
          py::arg("args"), // <--- 确保这里有正确的括号和结尾
          py::arg("mode") // <-- 新增mode参数的绑定
    );

    // ====================== 在这里新增绑定 ======================
        m.def("find_best_action", &find_best_action_for_state,
              "Finds the best action for a given board state using C++ MCTS engine.",
              py::arg("board_pieces"),
              py::arg("board_territory"),
              py::arg("current_player"),
              py::arg("current_move_number"),
              py::arg("model_path"),
              py::arg("use_gpu"),
              py::arg("args")
        );
        // ========================================================
}