# file: coach.py (ABSOLUTE FINAL VERSION - Corrected Args)
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
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

from neural_net import ExtendedConnectNet
from config import args
import cpp_mcts_engine

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

def inference_server_func(model, device, job_q, result_q, stop_event, batch_size, board_size):
    model.eval()
    with torch.no_grad():
        while not stop_event.is_set():
            # ==================== 从这里开始替换 ====================

            state_batch, id_batch = [], []

            # 1. 阻塞式等待，直到获取到批次的第一个任务
            try:
                # 可以设置一个较长的超时，比如1秒。如果1秒都没有任务，可能就真的没事干了
                req_id, state = job_q.get(timeout=1.0)
                state_batch.append(state)
                id_batch.append(req_id)
            except queue.Empty:
                # 如果长时间没有任务，则继续外层循环
                continue

            # 2. 第一个任务已收到，现在快速将队列中“已经存在”的其他任务也扫进批次
            #    直到批次满，或者队列变空
            while len(id_batch) < batch_size:
                try:
                    # 使用get_nowait()或get(block=False)，它不会等待，队列为空则立即抛出异常
                    req_id, state = job_q.get_nowait()
                    state_batch.append(state)
                    id_batch.append(req_id)
                except queue.Empty:
                    # 队列已空，说明我们已将所有积压的任务都收集了，可以跳出循环去处理批次
                    break

            # ==================== 到这里替换结束 ====================

            # 后续的代码保持不变
            # if not state_batch: continue ...
            # state_tensor = ...
            # log_policies, values = model(state_tensor) ...
            if not state_batch:
                continue

            state_tensor = torch.tensor(np.array(state_batch), device=device, dtype=torch.float32)
            state_tensor = state_tensor.view(-1, 6, board_size, board_size)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                log_policies, values = model(state_tensor)

            policies = torch.exp(log_policies).cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()

            for req_id, policy, value in zip(id_batch, policies, values):
                result_q.put((req_id, policy.tolist(), float(value)))

class Coach:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'], weight_decay=0.0001)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args['num_epochs'])
        self.scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.training_data = []

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
            print("步骤1：启动C++引擎进行自我对弈 (此过程将阻塞，请耐心等待)...")
            job_queue = queue.Queue(maxsize=self.args['batch_size'] * 2)
            result_queue = queue.Queue()
            final_data_queue = queue.Queue()
            stop_event = threading.Event()
            server_thread = threading.Thread(
                target=inference_server_func,
                args=(self.model, device, job_queue, result_queue, stop_event, self.args['batch_size'], self.args['board_size'])
            )
            server_thread.start()

            # =================== 关键修正在这里 ===================
            # 创建C++代码真正需要的参数字典
            cpp_args = {
                'num_selfPlay_episodes': self.args['num_selfPlay_episodes'], # C++用这个来决定总任务数
                'num_cpu_threads': self.args['num_cpu_threads'],             # C++用这个来决定线程池大小
                'num_searches': self.args['num_searches']                    # C++ MCTS的模拟次数
            }
            # ====================================================

            cpp_mcts_engine.run_parallel_self_play(job_queue, result_queue, final_data_queue, cpp_args)

            stop_event.set()
            server_thread.join()

            print("\n自我对弈完成！正在收集数据...")
            with tqdm.tqdm(total=self.args['num_selfPlay_episodes'], desc="收集数据") as pbar:
                while pbar.n < self.args['num_selfPlay_episodes']:
                    try:
                        result = final_data_queue.get_nowait()
                        if result.get("type") == "data":
                            self.training_data.extend(result.get("data", []))
                            pbar.update(1)
                    except queue.Empty:
                        break

            print(f"\n经验库大小: {len(self.training_data)}")
            self.training_data = self.training_data[-self.args['data_max_size']:]

            print("\n步骤2：训练神经网络 (使用GPU)...")
            if not self.training_data:
                print("经验库为空，跳过本轮训练。")
            else:
                self.model.train()
                for _ in tqdm.tqdm(range(self.args['num_epochs']), desc="训练模型"):
                    self.train()

            self.scheduler.step()
            torch.save(self.model.state_dict(), f"model_{i}.pth")
            print(f"模型 model_{i}.pth 已保存。")
        print(f"\n训练完成！")

    def evaluate_models(self, model1_path, model2_path):
        print(f"\n------ 开始评估 (C++ 引擎驱动) ------")
        if not model1_path or not model2_path:
            print("缺少模型文件，跳过评估。")
            return
        device_eval = torch.device("cpu")
        try:
            model1 = ExtendedConnectNet(board_size=self.args['board_size'], num_res_blocks=self.args['num_res_blocks'], num_hidden=self.args['num_hidden']).to(device_eval)
            model1.load_state_dict(torch.load(model1_path, map_location=device_eval))
            model1.eval()
            model2 = ExtendedConnectNet(board_size=self.args['board_size'], num_res_blocks=self.args['num_res_blocks'], num_hidden=self.args['num_hidden']).to(device_eval)
            model2.load_state_dict(torch.load(model2_path, map_location=device_eval))
            model2.eval()
        except Exception as e:
            print(f"加载评估模型失败: {e}, 跳过评估。")
            return

        scores = {1: 0, -1: 0, 0: 0}
        eval_args = {
            'num_eval_simulations': self.args.get('num_eval_simulations', 20),
            'board_size': self.args['board_size']
        }
        for i in tqdm.tqdm(range(self.args['num_eval_games']), desc="评估对战"):
            p1 = model2 if i % 2 == 0 else model1
            p2 = model1 if i % 2 == 0 else model2
            winner = cpp_mcts_engine.play_game_for_eval(p1, p2, eval_args)
            if winner == 1: scores[1 if i % 2 == 0 else -1] += 1
            elif winner == -1: scores[-1 if i % 2 == 0 else 1] += 1
            else: scores[0] += 1

        win_rate = scores[1] / self.args['num_eval_games'] if self.args['num_eval_games'] > 0 else 0
        print("\n------ 评估结果 ------")
        print(f"总对局数: {self.args['num_eval_games']}")
        print(f"新模型 vs 旧模型 (胜/负/平): {scores[1]} / {scores[-1]} / {scores[0]}")
        print(f"新模型胜率: {win_rate:.2%}")
        if win_rate > self.args.get('eval_win_rate', 0.55):
            print("结论：新模型有显著提升！👍")
        else:
            print("结论：新模型提升不明显或没有提升。")

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
        print("未找到任何已有模型，将从第 1 轮开始全新训练。")
    coach = Coach(model, args)
    coach.learn(start_epoch=start_epoch)
    model_after_training, _ = find_latest_model_file()
    if model_before_training and model_after_training != model_before_training:
        coach.evaluate_models(model_before_training, model_after_training)
    else:
        print("\n未进行有效的新一轮训练或未找到旧模型，跳过评估。")