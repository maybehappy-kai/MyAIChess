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

# =================== 这里是关键修改 ===================
ext_modules = [
    torch.utils.cpp_extension.CppExtension(
        'cpp_mcts_engine',
        [
            'cpp_src/bindings.cpp',
            'cpp_src/SelfPlayManager.cpp',
            'cpp_src/Gomoku.cpp',  # <-- 新增：告诉编译器编译Gomoku的实现
            'cpp_src/Node.cpp'  # <-- 新增：告诉编译器编译Node的实现
            # 'cpp_src/mcts.cpp' 已经被废弃，可以从列表中移除
        ],
        # In setup.py
        extra_compile_args={
            'cxx': [],  # <-- 恢复为空列表
        },
        extra_link_args=[
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
