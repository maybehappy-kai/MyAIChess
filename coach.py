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
            job_queue = queue.Queue() # maxsize不再需要，因为每次只放一个大的工作包
            result_queue = queue.Queue()
            final_data_queue = queue.Queue()
            stop_event = threading.Event()

            # 注意：新的inference_server_func不再需要batch_size参数
            server_thread = threading.Thread(
                target=inference_server_func,
                args=(self.model, device, job_queue, result_queue, stop_event, self.args['board_size'])
            )
            server_thread.start()

            cpp_args = {
                'num_selfPlay_episodes': self.args['num_selfPlay_episodes'],
                'num_cpu_threads': self.args['num_cpu_threads'],
                'num_searches': self.args['num_searches']
            }

            cpp_mcts_engine.run_parallel_self_play(job_queue, result_queue, final_data_queue, cpp_args)

            stop_event.set()
            server_thread.join()

            print("\n自我对弈完成！正在收集数据...")
            # 这个数据收集逻辑是正确的
            with tqdm.tqdm(total=self.args['num_selfPlay_episodes'], desc="收集数据") as pbar:
                games_processed = 0
                while games_processed < self.args['num_selfPlay_episodes']:
                    try:
                        result = final_data_queue.get(timeout=1.0) # 加一个超时以防万一
                        if result.get("type") == "data":
                            self.training_data.extend(result.get("data", []))
                            games_processed += 1
                            pbar.update(1)
                    except queue.Empty:
                        print("\n警告：数据队列为空，但自我对弈已结束。可能某些对局未能生成数据。")
                        break # 如果队列空了，就跳出循环

            print(f"\n经验库大小: {len(self.training_data)}")
            if len(self.training_data) > self.args['data_max_size']:
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

    # evaluate_models 函数无需改动
    def evaluate_models(self, model1_path, model2_path):
        print(f"\n------ 开始评估 (C++ 引擎驱动) ------")
        if not model1_path or not model2_path:
            print("缺少模型文件，跳过评估。")
            return
        device_eval = torch.device("cpu")
        try:
            model1 = ExtendedConnectNet(board_size=self.args['board_size'], num_res_blocks=self.args['num_res_blocks'],
                                        num_hidden=self.args['num_hidden']).to(device_eval)
            model1.load_state_dict(torch.load(model1_path, map_location=device_eval))
            model1.eval()
            model2 = ExtendedConnectNet(board_size=self.args['board_size'], num_res_blocks=self.args['num_res_blocks'],
                                        num_hidden=self.args['num_hidden']).to(device_eval)
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
            if winner == 1:
                scores[1 if i % 2 == 0 else -1] += 1
            elif winner == -1:
                scores[-1 if i % 2 == 0 else 1] += 1
            else:
                scores[0] += 1

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