# file: setup.py
from setuptools import setup, Extension
import torch.utils.cpp_extension
import sys
import os

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
            # 确保列表中的每个文件都是一个独立的字符串
            'cpp_src/bindings.cpp',
            'cpp_src/SelfPlayManager.cpp',
            'cpp_src/Gomoku.cpp',
            'cpp_src/Node.cpp',
            'cpp_src/InferenceEngine.cpp'  # <--- 这一行很可能是您出错的地方，请确保它和上一行有逗号隔开
        ],
        extra_compile_args={
            'cxx': ['/Zi', '/Od', '/FS'],
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