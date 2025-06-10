#pragma once

// 再次引入pybind11的核心头文件
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // <<< 新增的这一行

// 为pybind11的命名空间创建一个简短的别名 py
namespace py = pybind11;

/*
 * 这是我们要在C++中实现的核心加速函数。
 * 它的作用是接收Python端的一个MCTS“根节点”对象，
 * 然后在C++中执行指定次数的高性能循环搜索。
 * 在搜索过程中，它会回调Python端的模型进行神经网络推断。
 *
 * 参数:
 * root_node: 一个Python对象，代表MCTS树的根节点。
 * model:     Python端的神经网络模型对象。
 * args:      一个Python字典，包含了所有的超参数 (如num_searches)。
*/
void run_mcts_simulations(const py::object& root_node, const py::object& model, const py::dict& args);