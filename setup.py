# file: setup.py (修正版)
from setuptools import setup, Extension
import torch.utils.cpp_extension
import sys
import os
import shutil

# --- 清理旧的构建目录 ---
build_dir = "build"
if os.path.exists(build_dir):
    print(f"正在清理旧的构建目录: {build_dir}")
    shutil.rmtree(build_dir)
# --------------------------

# --- Python环境信息部分保持不变 ---
py_major = sys.version_info.major
py_minor = sys.version_info.minor
python_lib_name = f"python{py_major}{py_minor}"
python_lib_dir = os.path.join(os.path.dirname(sys.executable), 'libs')

# =================== 核心修正区域 ===================
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
        extra_compile_args={
            # 在Windows (MSVC)下, 使用'/std:c++17'来指定C++标准
            # '/Zi' 用于生成调试信息, '/Od' 用于关闭优化, '/FS' 是VS中的一个标准标志
            'cxx': ['/Zi', '/Od', '/FS', '/std:c++17'],
        },
        extra_link_args=[
            '/DEBUG',
            f"/LIBPATH:{python_lib_dir}",
            f"{python_lib_name}.lib"
        ]
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