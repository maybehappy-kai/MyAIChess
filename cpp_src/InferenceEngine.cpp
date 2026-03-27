#include "InferenceEngine.h"
#include <cmath>     // for std::sqrt
#include <stdexcept> // for std::runtime_error
#include <string>

// 构造函数的实现
InferenceEngine::InferenceEngine(const std::string &model_path, bool use_gpu)
    : device_(use_gpu ? torch::kCUDA : torch::kCPU)
{ // 根据标志设置设备
    try
    {
        // 使用 torch::jit::load 加载您导出的 .pt 文件
        module_ = torch::jit::load(model_path);
        // 将模型移动到指定的设备（CPU或GPU）
        module_.to(device_);

        if (use_gpu)
        { // 半精度优化通常只在GPU上进行
            module_.to(torch::kHalf);
        }
        // 切换到评估模式，这会关闭Dropout等训练特有的层
        module_.eval();
    }
    catch (const c10::Error &e)
    {
        // 如果模型加载失败，抛出一个清晰的异常
        throw std::runtime_error("Failed to load TorchScript model: " + std::string(e.what()));
    }
}

InferenceEngine::InferenceResult InferenceEngine::infer(const std::vector<std::vector<float>> &batch_states, int board_size, int num_channels)
{
    if (batch_states.empty())
    {
        return {};
    }

    if (board_size <= 0 || num_channels <= 0)
    {
        throw std::runtime_error("InferenceEngine::infer received invalid board_size or num_channels.");
    }

    const size_t expected_state_size =
        static_cast<size_t>(board_size) *
        static_cast<size_t>(board_size) *
        static_cast<size_t>(num_channels);
    if (expected_state_size == 0)
    {
        throw std::runtime_error("InferenceEngine::infer calculated an empty expected state size.");
    }

    std::vector<float> flat_batch;
    size_t total_elements = batch_states.size() * expected_state_size;
    flat_batch.reserve(total_elements);
    for (size_t i = 0; i < batch_states.size(); ++i)
    {
        const auto &state = batch_states[i];
        if (state.size() != expected_state_size)
        {
            throw std::runtime_error(
                "InferenceEngine::infer state shape mismatch at index " + std::to_string(i) +
                ": expected " + std::to_string(expected_state_size) +
                ", got " + std::to_string(state.size()));
        }
        flat_batch.insert(flat_batch.end(), state.begin(), state.end());
    }

    torch::Tensor input_tensor = torch::from_blob(
        flat_batch.data(),
        {static_cast<long>(batch_states.size()), static_cast<long>(expected_state_size)},
        torch::kFloat32);
    input_tensor = input_tensor.clone().to(device_);

    if (device_.is_cuda())
    { // 同样只在GPU上转换
        input_tensor = input_tensor.to(torch::kHalf);
    }

    input_tensor = input_tensor.view({-1, num_channels, board_size, board_size});

    // ... (后续的前向传播、解析结果等代码完全保持不变) ...
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    at::NoGradGuard no_grad;
    auto output_tuple = module_.forward(inputs).toTuple();

    torch::Tensor policy_log_probs = output_tuple->elements()[0].toTensor().to(torch::kCPU).to(torch::kFloat);
    torch::Tensor values = output_tuple->elements()[1].toTensor().to(torch::kCPU).to(torch::kFloat);

    auto policies = torch::exp(policy_log_probs);

    std::vector<std::vector<float>> policy_vec;
    policy_vec.reserve(policies.size(0));
    for (int i = 0; i < policies.size(0); ++i)
    {
        std::vector<float> p(policies[i].data_ptr<float>(), policies[i].data_ptr<float>() + policies[i].numel());
        policy_vec.push_back(p);
    }

    std::vector<float> value_vec(values.data_ptr<float>(), values.data_ptr<float>() + values.numel());

    return {policy_vec, value_vec};
}