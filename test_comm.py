# file: test_comm.py
import os
import torch
import queue
import threading
import time
import cpp_mcts_engine
import numpy as np # 需要numpy来处理

try:
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    os.add_dll_directory(torch_lib_path)
except (AttributeError, FileNotFoundError):
    pass

# 新的服务器，模拟神经网络的行为
def server_func(job_q, result_q, stop_event):
    print("[Python Server] Started.")
    board_size = 9
    action_size = board_size * board_size

    while not stop_event.is_set():
        try:
            request_id, state = job_q.get(timeout=0.1)

            # 打印收到的状态信息（可选，用于调试）
            # state现在是一个浮点数列表
            print(f"[Python Server] Received job {request_id} with state length {len(state)}")

            # 模拟神经网络的输出
            # 创建一个均匀分布的虚拟策略向量
            dummy_policy = np.ones(action_size, dtype=np.float32) / action_size
            # 创建一个虚拟的价值
            dummy_value = np.random.uniform(-1.0, 1.0)

            # 将numpy数组转为列表发送回C++
            result_q.put((request_id, dummy_policy.tolist(), dummy_value))

        except queue.Empty:
            continue
    print("[Python Server] Shutting down.")


if __name__ == '__main__':
    # 这里用标准库的queue，因为它在同一个进程的线程间通信更高效
    job_queue = queue.Queue()
    result_queue = queue.Queue()
    final_data_queue = queue.Queue()

    # 这里的参数需要和coach.py里的args保持一致
    args = {
        'num_workers': 4,
        'num_searches': 25 # MCTS模拟次数，可以设小一点方便测试
    }

    stop_event = threading.Event()
    server_thread = threading.Thread(
        target=server_func,
        args=(job_queue, result_queue, stop_event)
    )
    server_thread.start()

    print("[Python Main] Calling C++ engine to play games...")
    start_time = time.time()

    cpp_mcts_engine.run_parallel_self_play(
        job_queue=job_queue,
        result_queue=result_queue,
        final_data_queue=final_data_queue,
        args=args
    )

    end_time = time.time()
    print(f"[Python Main] C++ engine finished in {end_time - start_time:.2f} seconds.")

    stop_event.set()
    server_thread.join()

    # 检查C++是否成功返回了训练数据
    num_games_finished = 0
    while not final_data_queue.empty():
        try:
            data = final_data_queue.get_nowait()
            if data.get("type") == "data":
                num_games_finished += 1
                print(f"Received training data for one game. Total examples: {len(data['data'])}")
        except queue.Empty:
            break

    print(f"\nSUCCESS: Engine completed {num_games_finished} games!")