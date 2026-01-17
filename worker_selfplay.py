# file: worker_selfplay.py
# 最终完整版：包含 Windows 内存清理、动态模型加载、异常恢复
import torch
import os
import time
import queue
import pickle
import uuid
import re
import platform
import ctypes
import cpp_mcts_engine
from config import args

# --- 配置区域 ---
SELFPLAY_DEVICE_ID = 0  # 显卡 0
DATA_BUFFER_DIR = "data_buffer"


# ----------------

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def clear_windows_memory():
    """保留原项目的 Windows 内存清理功能"""
    if platform.system() == "Windows":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass


def find_latest_model_pt():
    path = "."
    max_epoch = -1
    latest_file = None
    pattern = re.compile(r"model_(\d+)_.*\.pt$")

    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = f
    return latest_file, max_epoch


def main():
    # 强制只使用第一张显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(SELFPLAY_DEVICE_ID)
    ensure_dir(DATA_BUFFER_DIR)

    # >>>>>>>>【新增】核心修复代码 >>>>>>>>
    args['num_channels'] = (args.get('history_steps', 0) + 1) * 4 + 4
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    print(f"--- 启动自对弈 Worker (GPU {SELFPLAY_DEVICE_ID}) ---")
    print(f"数据缓冲区: {DATA_BUFFER_DIR}/")

    last_loaded_model = None

    while True:
        try:
            # 1. 动态寻找最新模型
            model_file, epoch = find_latest_model_pt()

            if model_file is None:
                print("\r等待初始模型 (model_0.pt)...", end="")
                time.sleep(5)
                continue

            if model_file != last_loaded_model:
                print(f"\n[切换模型] 发现新版本: {model_file} (Epoch {epoch})")
                last_loaded_model = model_file
                # 切换模型时清理一次内存
                clear_windows_memory()

            # 2. 执行自对弈
            data_queue = queue.Queue()
            current_args = args.copy()

            # 注意：这里我们不像原版那样在失败时减少局数
            # 因为并行架构下，只要没新模型，我们就应该全力为当前模型生成数据
            print(f"正在生成数据 (Base: {model_file})... ", end="", flush=True)

            start_time = time.time()
            cpp_mcts_engine.run_parallel_self_play(
                model_file,
                True,  # use_gpu
                data_queue,
                current_args
            )
            duration = time.time() - start_time

            # 3. 数据落盘
            new_data = []
            while not data_queue.empty():
                item = data_queue.get()
                if item.get("type") == "data":
                    new_data.extend(item.get("data", []))

            if len(new_data) > 0:
                filename = f"batch_{epoch}_{int(time.time())}_{str(uuid.uuid4())[:8]}.pkl"
                save_path = os.path.join(DATA_BUFFER_DIR, filename)

                with open(save_path, 'wb') as f:
                    pickle.dump(new_data, f)
                print(f"完成! (+{len(new_data)} samples, {duration:.1f}s)")
            else:
                print("本轮无数据生成 (可能被异常中断)")

            # 定期清理内存
            clear_windows_memory()

        except Exception as e:
            print(f"\n[错误] 自对弈循环发生异常: {e}")
            print("5秒后重试...")
            time.sleep(5)


if __name__ == "__main__":
    main()