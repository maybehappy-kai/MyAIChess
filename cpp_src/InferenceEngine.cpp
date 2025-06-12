#include "InferenceEngine.h"
#include <cmath> // for std::sqrt
#include <stdexcept> // for std::runtime_error

// 构造函数的实现
InferenceEngine::InferenceEngine(const std::string& model_path, bool use_gpu)
    : device_(use_gpu ? torch::kCUDA : torch::kCPU) { // 根据标志设置设备
    try {
        // 使用 torch::jit::load 加载您导出的 .pt 文件
        module_ = torch::jit::load(model_path);
        // 将模型移动到指定的设备（CPU或GPU）
        module_.to(device_);
        // 切换到评估模式，这会关闭Dropout等训练特有的层
        module_.eval();
    } catch (const c10::Error& e) {
        // 如果模型加载失败，抛出一个清晰的异常
        throw std::runtime_error("Failed to load TorchScript model: " + std::string(e.what()));
    }
}

// 将整个 infer 函数替换为以下代码
InferenceEngine::InferenceResult InferenceEngine::infer(const std::vector<std::vector<float>>& batch_states) {
    if (batch_states.empty()) {
        return {};
    }

    // --- vvv 这里是修正的核心逻辑 vvv ---

    // 1. 手动将嵌套的vector扁平化 (flatten)
    std::vector<float> flat_batch;
    // 计算所有元素总数并预分配内存，这是最高效的方式
    size_t total_elements = batch_states.size() * batch_states[0].size();
    flat_batch.reserve(total_elements);

    // 遍历每一个state，将其追加到扁平化的vector中
    for (const auto& state : batch_states) {
        flat_batch.insert(flat_batch.end(), state.begin(), state.end());
    }

    // 2. 从扁平化的数据创建Tensor
    // torch::from_blob 可以高效地从已有的C++内存创建Tensor，避免不必要的数据拷贝
    torch::Tensor input_tensor = torch::from_blob(
        flat_batch.data(), // 指向数据块的指针
        {static_cast<long>(batch_states.size()), static_cast<long>(batch_states[0].size())}, // 初始形状
        torch::kFloat32
    );

    // .clone() 会为Tensor创建一份自己的数据拷贝，确保在flat_batch生命周期结束后数据依然有效
    input_tensor = input_tensor.clone().to(device_);

    // --- ^^^ 修正结束 ^^^ ---

    // 3. 将Tensor重塑为模型期望的形状 [N, C, H, W] (这部分逻辑和之前一样)
    const int board_size = static_cast<int>(std::sqrt(batch_states[0].size() / 6));
    input_tensor = input_tensor.view({-1, 6, board_size, board_size});

    // ... (后面的前向传播、解析结果等代码完全保持不变) ...
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    at::NoGradGuard no_grad;
    auto output_tuple = module_.forward(inputs).toTuple();

    torch::Tensor policy_log_probs = output_tuple->elements()[0].toTensor().to(torch::kCPU);
    torch::Tensor values = output_tuple->elements()[1].toTensor().to(torch::kCPU);

    auto policies = torch::exp(policy_log_probs);

    std::vector<std::vector<float>> policy_vec;
    policy_vec.reserve(policies.size(0));
    for (int i = 0; i < policies.size(0); ++i) {
        std::vector<float> p(policies[i].data_ptr<float>(), policies[i].data_ptr<float>() + policies[i].numel());
        policy_vec.push_back(p);
    }

    std::vector<float> value_vec(values.data_ptr<float>(), values.data_ptr<float>() + values.numel());

    return {policy_vec, value_vec};
}