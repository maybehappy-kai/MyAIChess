#include "mcts.h" // 引入我们自己的头文件
#include <torch/extension.h> // PyTorch官方提供的C++扩展头文件，包含了与Tensor交互的很多功能

// 再次为pybind11的命名空间创建别名
namespace py = pybind11;

// 这是我们核心加速函数的具体实现
void run_mcts_simulations(const py::object& root_node, const py::object& model, const py::dict& args) {
    // 从Python字典中获取num_searches参数，并转换为C++的int类型
    int num_searches = args["num_searches"].cast<int>();

    // --- 这就是我们要从Python中剥离出来的、最耗时的循环 ---
    for (int i = 0; i < num_searches; ++i) {
        py::gil_scoped_acquire acquire;
        py::object node = root_node;

        // 1. 选择 (Selection)
        // 只要节点完全扩展了，就继续向下选择最优子节点
        // 我们通过 .attr("method_name")() 的方式在C++中调用Python对象的方法
        while (node.attr("is_fully_expanded")().cast<bool>()) {
            node = node.attr("select")();
        }

        // 2. 扩展 (Expansion) 与 模拟 (Simulation)
        // 获取叶子节点的游戏状态
        py::object game = node.attr("game");
        py::tuple game_status = game.attr("get_game_ended")().cast<py::tuple>();
        double value = game_status[0].cast<double>();
        bool is_terminal = game_status[1].cast<bool>();

        if (!is_terminal) {
            // 如果游戏没有结束，我们需要请求神经网络进行一次推断

            // 从Python端的模型对象获取它所在的设备（'cpu'或'cuda'）
            py::object model_device = model.attr("device");

            // 获取当前游戏状态的numpy数组，并转换为torch.Tensor
            py::array state_np = game.attr("get_state")().cast<py::array>();
            py::object torch_module = py::module_::import("torch");
            py::object state_tensor = torch_module.attr("tensor")(state_np)
                                          .attr("unsqueeze")(0)
                                          .attr("to")(model_device, torch::kFloat32);

            // 在C++中调用Python模型的forward方法
            py::tuple result = model(state_tensor);

            // 【关键】使用完Python对象后，立刻释放GIL，让其他Python线程可以工作
            // py::gil_scoped_release release;

            // 从返回的元组中解析出策略和价值
            py::array policy_np = result[0].attr("exp")().attr("squeeze")(0).attr("cpu")().attr("numpy")().cast<py::array>();
            value = result[1].attr("item")().cast<double>();

            // 调用Python端的Node对象的expand方法，用新的策略来扩展节点
            node.attr("expand")(policy_np);
        }

        // 3. 反向传播 (Backpropagation)
        // 调用Python端的Node对象的backpropagate方法，将价值传回去
        node.attr("backpropagate")(value);
    }
}