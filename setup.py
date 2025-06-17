# file: setup.py (最终修正版)
from setuptools import setup, Extension
import torch.utils.cpp_extension
import sys
import os

# --- Python环境信息部分保持不变 ---
py_major = sys.version_info.major
py_minor = sys.version_info.minor

# =================== 核心修正区域 ===================
# 移除所有调试相关的编译和链接参数
# 对于Release版本，通常让编译器自己选择最优优化（如 /O2）
# 我们这里留空，使用 setuptools 和 PyTorch 的默认优化配置
ext_modules = [
    torch.utils.cpp_extension.CppExtension(
        'cpp_mcts_engine',
        [
            'cpp_src/bindings.cpp',
            'cpp_src/SelfPlayManager.cpp',
            'cpp_src/Gomoku.cpp',
            'cpp_src/Node.cpp',
            'cpp_src/InferenceEngine.cpp'
        ],
        extra_compile_args={'cxx': []}, # <-- 清空编译参数
        extra_link_args=[] # <-- 清空链接参数
    )
]
# ====================================================

setup(
    name='cpp_mcts_engine',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': torch.utils.cpp_extension.BuildExtension
    }
)