#pragma once

#include <torch/script.h> // 引用 LibTorch 的核心头文件
#include <vector>
#include <string>
#include <utility> // for std::pair

class InferenceEngine
{
public:
    // 为推理结果定义一个清晰的类型别名，包含一个策略向量的批次和一个价值向量
    using InferenceResult = std::pair<std::vector<std::vector<float>>, std::vector<float>>;

    // 构造函数：接收模型文件路径和是否使用GPU的标志
    InferenceEngine(const std::string &model_path, bool use_gpu);

    // 核心功能：接收一批游戏状态，返回一批推理结果
    InferenceResult infer(const std::vector<std::vector<float>> &batch_states, int board_size, int num_channels);

private:
    torch::jit::script::Module module_; // 用于存储加载的TorchScript模型
    torch::Device device_;              // 存储设备类型 (CPU 或 CUDA)
};