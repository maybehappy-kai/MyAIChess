# file: coach.py (FINAL, BATCHED MCTS VERSION)
import torch
import torch.nn.functional as F
import torch.optim as optim
import queue
import threading
import numpy as np
import tqdm
import random
import os
import re
from torch.optim.lr_scheduler import CosineAnnealingLR

from neural_net import ExtendedConnectNet
from config import args
import cpp_mcts_engine
from collections import deque  # <-- 新增这一行
import platform
import ctypes


# 定义一个仅在Windows下生效的内存清理函数
def clear_windows_memory():
    if platform.system() == "Windows":
        try:
            # 调用Windows API来强制清理当前进程的内存工作集
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            print("[System] Windows memory working set has been cleared.")
        except Exception as e:
            print(f"[System] Failed to clear Windows memory working set: {e}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_model_file():
    path = "."
    max_epoch = 0
    latest_file = None
    pattern = re.compile(r"model_(\d+)\.pth")
    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = f
    start_epoch = max_epoch + 1 if latest_file else 1
    return latest_file, start_epoch


# ==================== 这是唯一需要修改的地方 ====================
# 新的、为批处理MCTS优化的推理服务器
def inference_server_func(model, device, job_q, result_q, stop_event, board_size):
    model.eval()
    with torch.no_grad():
        while not stop_event.is_set():
            try:
                # 1. 一次只获取一个“工作包”，这个包里包含了整个批次
                #    设置一个超时，以便能定期检查 stop_event
                request_id, state_batch = job_q.get(timeout=1.0)

                # 2. state_batch 现在是一个状态列表，直接转换成numpy数组
                state_tensor = torch.tensor(np.array(state_batch), device=device, dtype=torch.float32)

                # 确保张量形状正确 (B, C, H, W)
                state_tensor = state_tensor.view(-1, 6, board_size, board_size)

                # 使用自动混合精度进行推理
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    log_policies, values = model(state_tensor)

                # 将torch张量转为Python列表，方便后续处理
                # C++端将接收到 [[p1,p2...], [p1,p2...]] 和 [v1, v2...]
                policies = torch.exp(log_policies).cpu().numpy().tolist()
                values = values.squeeze(-1).cpu().numpy().tolist()

                # 3. 将整个批次的结果打包后，一次性放回结果队列
                result_q.put((request_id, policies, values))

            except queue.Empty:
                # 队列为空是正常现象，继续循环，检查stop_event
                continue


# =============================================================


class Coach:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'], weight_decay=0.0001)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args['num_epochs'])
        self.scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.training_data = deque(maxlen=self.args['data_max_size'])  # <-- 修改这一行

    def train(self):
        self.model.train()
        if len(self.training_data) < self.args['batch_size']: return
        batch = random.sample(self.training_data, self.args['batch_size'])
        states, target_policies, target_values = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        target_policies = torch.tensor(np.array(target_policies), dtype=torch.float32).to(device)
        target_values = torch.tensor(np.array(target_values), dtype=torch.float32).unsqueeze(1).to(device)
        states = states.view(-1, 6, self.args['board_size'], self.args['board_size'])
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            pred_log_policies, pred_values = self.model(states)
            policy_loss = -torch.sum(target_policies * pred_log_policies) / len(target_policies)
            value_loss = F.mse_loss(pred_values, target_values)
            total_loss = policy_loss + self.args['value_loss_weight'] * value_loss
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def learn(self, start_epoch=1):
        for i in range(start_epoch, start_epoch + self.args['num_iterations']):
            print(f"------ 迭代轮次: {i} ------")
            # --- vvv 这里是需要修改的C++引擎调用部分 vvv ---

            # --- vvv 这里是需要修改的核心区域 vvv ---
            print("步骤1：启动纯C++引擎进行自我对弈 (此过程将阻塞)...")
            final_data_queue = queue.Queue()

            cpp_args = {
                'num_selfPlay_episodes': self.args['num_selfPlay_episodes'],
                'num_cpu_threads': self.args['num_cpu_threads'],
                'num_searches': self.args['num_searches']
            }

            # 【修改】获取前一轮的模型路径，而不是当前轮次的
            model_path_pt = f"model_{i - 1}.pt"
            use_gpu = (device.type == 'cuda')

            print(f"[Python Coach] 指示C++引擎使用模型: {model_path_pt}")
            cpp_mcts_engine.run_parallel_self_play(
                model_path_pt,
                use_gpu,
                final_data_queue,
                cpp_args
            )
            # --- ^^^ 修改结束 ^^^ ---

            # 5. 不再需要停止和加入推理线程
            # stop_event.set()
            # server_thread.join()

            # --- ^^^ 修改结束 ^^^ ---

            print("\n自我对弈完成！正在收集数据...")
            # 这个数据收集逻辑是正确的
            with tqdm.tqdm(total=self.args['num_selfPlay_episodes'], desc="收集数据") as pbar:
                games_processed = 0
                while games_processed < self.args['num_selfPlay_episodes']:
                    try:
                        result = final_data_queue.get(timeout=1.0)  # 加一个超时以防万一
                        if result.get("type") == "data":
                            self.training_data.extend(result.get("data", []))
                            games_processed += 1
                            pbar.update(1)
                    except queue.Empty:
                        print("\n警告：数据队列为空，但自我对弈已结束。可能某些对局未能生成数据。")
                        break  # 如果队列空了，就跳出循环

            print(f"\n经验库大小: {len(self.training_data)}")
            # if len(self.training_data) > self.args['data_max_size']:
            # self.training_data = self.training_data[-self.args['data_max_size']:]

            print("\n步骤2：训练神经网络 (使用GPU)...")
            if not self.training_data:
                print("经验库为空，跳过本轮训练。")
            else:
                self.model.train()
                for _ in tqdm.tqdm(range(self.args['num_epochs']), desc="训练模型"):
                    self.train()

            self.scheduler.step()
            # --- vvv 这里是您需要修改的核心区域 vvv ---

            # 1. 保存标准的 .pth 权重文件 (这行代码保持不变)
            model_path_pth = f"model_{i}.pth"
            torch.save(self.model.state_dict(), model_path_pth)
            print(f"模型 {model_path_pth} 已保存。")

            # 2. 【新增代码】导出可供C++使用的 TorchScript 模型
            model_path_pt = f"model_{i}.pt"
            self.model.eval()  # 导出前，必须将模型切换到评估模式

            # 创建一个符合模型输入的示例张量。
            # 形状为 (batch_size, channels, height, width)
            example_input = torch.rand(
                1,
                6,  # 根据您的Gomoku.cpp，状态有6个通道
                self.args['board_size'],
                self.args['board_size']
            ).to(device)

            try:
                # 使用 torch.jit.trace 功能追踪模型的计算图
                traced_script_module = torch.jit.trace(self.model, example_input)
                # 将追踪到的计算图保存为 .pt 文件
                traced_script_module.save(model_path_pt)
                print(f"TorchScript模型 {model_path_pt} 已成功导出，可供C++使用。")
            except Exception as e:
                print(f"【错误】导出TorchScript模型失败: {e}")

            # --- ^^^ 修改结束 ^^^ ---
            clear_windows_memory()
        print(f"\n训练完成！")

    # evaluate_models 函数无需改动
    def evaluate_models(self, model1_path, model2_path):
        print(f"\n------ 开始分组诊断式评估 (C++ 引擎驱动) ------")
        if not model1_path or not os.path.exists(model1_path) or \
                not model2_path or not os.path.exists(model2_path):
            print("评估缺少必要的模型文件，跳过评估。")
            return

        model1_pt_path = model1_path.replace('.pth', '.pt')  # 旧模型
        model2_pt_path = model2_path.replace('.pth', '.pt')  # 新模型
        use_gpu = (device.type == 'cuda')

        total_games = self.args.get('num_eval_games', 50)
        games_per_side = total_games // 2
        if games_per_side == 0:
            print("评估局数过少，无法进行分组评估。")
            return

        print(f"评估模型 (旧): {model1_pt_path}")
        print(f"评估模型 (新): {model2_pt_path}")

        base_eval_args = {
            'num_eval_games': games_per_side,
            'num_eval_simulations': self.args.get('num_eval_simulations', 20),
            'num_cpu_threads': self.args.get('num_cpu_threads', 12)
        }

        # --- 实验一：新模型执先手 (Model 2) ---
        print(f"\n[实验一] 新模型执黑，进行 {games_per_side} 局...")
        results1 = cpp_mcts_engine.run_parallel_evaluation(
            model1_pt_path, model2_pt_path, use_gpu, base_eval_args, mode=2
        )
        new_as_p1_wins = results1.get("model2_wins", 0)
        old_as_p2_wins = results1.get("model1_wins", 0)
        draws1 = results1.get("draws", 0)

        # --- 实验二：旧模型执先手 (Model 1) ---
        print(f"\n[实验二] 旧模型执黑，进行 {games_per_side} 局...")
        results2 = cpp_mcts_engine.run_parallel_evaluation(
            model1_pt_path, model2_pt_path, use_gpu, base_eval_args, mode=1
        )
        old_as_p1_wins = results2.get("model1_wins", 0)
        new_as_p2_wins = results2.get("model2_wins", 0)
        draws2 = results2.get("draws", 0)

        # --- 汇总和分析结果 ---
        total_new_wins = new_as_p1_wins + new_as_p2_wins
        total_old_wins = old_as_p1_wins + old_as_p2_wins
        total_draws = draws1 + draws2

        print("\n------ 诊断评估结果 ------")
        print(f"新模型执先手时，战绩 (新 vs 旧 | 胜/负/平): {new_as_p1_wins} / {old_as_p2_wins} / {draws1}")
        print(f"旧模型执先手时，战绩 (旧 vs 新 | 胜/负/平): {old_as_p1_wins} / {new_as_p2_wins} / {draws2}")
        print("---------------------------------")

        overall_win_rate = total_new_wins / (total_games) if total_games > 0 else 0
        print(f"综合战绩 - 新 vs 旧 (胜/负/平): {total_new_wins} / {total_old_wins} / {total_draws}")
        print(f"新模型综合胜率: {overall_win_rate:.2%}")

        if games_per_side > 0 and (new_as_p1_wins / games_per_side) > 0.9 and (old_as_p1_wins / games_per_side) > 0.9:
            print("\n【诊断结论】: AI已发现并掌握了 '先手必胜' 策略。")
            print("下一步建议：在自对弈中引入'狄利克雷噪声'和'温度采样'来打破僵局，探索后手获胜的可能性。")
        elif overall_win_rate > self.args.get('eval_win_rate', 0.52):
            print("\n【诊断结论】: 新模型有显著提升！👍")
        else:
            print("\n【诊断结论】: 新模型提升不明显或没有提升，可能陷入了局部最优。")


if __name__ == '__main__':
    print(f"将要使用的设备 (主进程/训练): {device}")
    model_before_training, start_epoch = find_latest_model_file()
    model = ExtendedConnectNet(
        board_size=args['board_size'],
        num_res_blocks=args['num_res_blocks'],
        num_hidden=args['num_hidden']
    ).to(device)

    if model_before_training:
        try:
            print(f"找到最新模型 {model_before_training}，将从第 {start_epoch} 轮开始继续训练...")
            model.load_state_dict(torch.load(model_before_training, map_location=device))
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}，将从零开始训练。")
            start_epoch = 1
            model_before_training = None
    else:
        # --- vvv 这里是需要修改的核心区域 vvv ---
        print("未找到任何已有模型，将从第 1 轮开始全新训练。")
        print("正在创建并保存初始随机模型 (model_0)...")

        # 【新增代码】为第一轮自我对弈准备一个“第0代”模型
        model.eval()
        example_input = torch.rand(1, 6, args['board_size'], args['board_size']).to(device)
        try:
            traced_script_module = torch.jit.trace(model, example_input)
            traced_script_module.save("model_0.pt")
            torch.save(model.state_dict(), "model_0.pth")
            print("初始模型 model_0.pt 和 model_0.pth 已保存。")
        except Exception as e:
            print(f"【错误】创建初始模型失败: {e}, 程序无法继续。")
            exit()  # 如果初始模型都创建失败，直接退出

        start_epoch = 1  # 确认从第1轮开始
        # --- ^^^ 修改结束 ^^^ ---

    coach = Coach(model, args)
    coach.learn(start_epoch=start_epoch)
    # ... (后续评估代码不变) ...
    model_after_training, _ = find_latest_model_file()
    if model_before_training and model_after_training != model_before_training:
        coach.evaluate_models(model_before_training, model_after_training)
    else:
        print("\n未进行有效的新一轮训练或未找到旧模型，跳过评估。")

        # ====================== 新增的核心清理代码 ======================
        # 在程序即将结束时，手动触发资源释放，避免退出时卡死

        print("\n训练全部完成，正在手动清理内存...")

        # 检查coach对象是否存在，并清空其内部的大数据结构
        if 'coach' in locals() and hasattr(coach, 'training_data'):
            coach.training_data.clear()
            print("经验回放池已清空。")

        # 手动调用Python的垃圾回收器
        import gc

        gc.collect()

        print("内存清理完成。程序即将退出。")
        # ===============================================================