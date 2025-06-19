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
optimized_compile_args = [
    '/O2',         # 开启最高级别的速度优化
    '/GL',         # 开启全程序优化 (编译时部分)，即LTO
    '/DNDEBUG',    # 禁用调试性的断言 (assert)
    '/fp:fast',    # 允许更快的浮点数运算
    '/arch:AVX2',  # <-- 新增：明确启用AVX2指令集以最大化SIMD并行计算
]

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
        # 将优化标志应用到C++编译器
        extra_compile_args={'cxx': optimized_compile_args},
        # 链接时也需要启用全程序优化 (链接时部分)
        extra_link_args=['/LTCG']
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