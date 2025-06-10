import torch
import os

# 获取 torch 模块的安装路径
pytorch_path = os.path.dirname(torch.__file__)

print(f"PyTorch installation path: {pytorch_path}")