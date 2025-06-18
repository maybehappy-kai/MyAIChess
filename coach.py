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


def get_augmented_data(state, policy, board_size):
    """
    对单个训练样本进行8种对称变换的数据增强。
    :param state: Numpy array, 形状为 (6, board_size, board_size)
    :param policy: Numpy array, 形状为 (board_size * board_size,)
    :param board_size: 棋盘大小
    :return: 一个包含8个 (state, policy) 元组的列表
    """
    augmented_data = []

    # 将策略向量变回二维，方便操作
    policy_2d = policy.reshape(board_size, board_size)

    # 8种对称变换
    for i in range(1, 5):  # 旋转 0, 90, 180, 270 度

        # 变换 State
        augmented_state = np.rot90(state, i, axes=(1, 2))
        # 变换 Policy
        augmented_policy_2d = np.rot90(policy_2d, i)

        # 原始旋转版本
        augmented_data.append((augmented_state.copy(), augmented_policy_2d.flatten()))

        # 水平翻转后再旋转的版本
        flipped_state = np.flip(augmented_state, axis=2)
        flipped_policy_2d = np.flip(augmented_policy_2d, axis=1)
        augmented_data.append((flipped_state.copy(), flipped_policy_2d.flatten()))

    return augmented_data


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


# ====================== 智能模型管理函数 (全新) ======================

def transfer_weights(new_model, path_to_old_weights):
    """
    将旧模型（通常是较小的模型）的权重加载到新模型中。
    只加载层名和权重形状都匹配的层。
    """
    print(f"--- 启动迁移学习，从 '{path_to_old_weights}' 加载权重 ---")
    old_state_dict = torch.load(path_to_old_weights, map_location=torch.device('cpu'))
    new_model_state_dict = new_model.state_dict()
    loaded_count = 0
    skipped_count = 0
    for name, param in old_state_dict.items():
        if name in new_model_state_dict and new_model_state_dict[name].shape == param.shape:
            new_model_state_dict[name].copy_(param)
            loaded_count += 1
        else:
            skipped_count += 1
    new_model.load_state_dict(new_model_state_dict)
    print(f"--- 迁移学习完成。成功迁移 {loaded_count} 个层，跳过 {skipped_count} 个不兼容层。 ---")
    return new_model


def save_model(model, epoch, args):
    """
    保存模型，并自动生成带结构信息的文件名 (同时保存 .pth 和 .pt)
    """
    # vvvvvv 从args获取通道数并构建新的文件名 vvvvvv
    num_channels = args['num_channels']
    base_filename = f"model_{epoch}_{args['num_res_blocks']}x{args['num_hidden']}_{num_channels}c"
    # ^^^^^^ 从args获取通道数并构建新的文件名 ^^^^^^
    model_path_pth = f"{base_filename}.pth"
    model_path_pt = f"{base_filename}.pt"

    torch.save(model.state_dict(), model_path_pth)
    print(f"模型 {model_path_pth} 已保存。")

    model.eval()
    # vvvvvv 使用args中的通道数创建示例输入 vvvvvv
    example_input = torch.rand(1, num_channels, args['board_size'], args['board_size']).to(device)
    # ^^^^^^ 使用args中的通道数创建示例输入 ^^^^^^
    try:
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(model_path_pt)
        print(f"TorchScript模型 {model_path_pt} 已成功导出。")
    except Exception as e:
        print(f"【错误】导出TorchScript模型失败: {e}")


# 在 coach.py 中，用下面的新函数替换旧的 find_latest_model_file 函数

def find_latest_model_file():
    """
    查找最新的模型文件，当轮次（epoch）相同时，选择最近被修改的文件。
    """
    path = "."
    max_epoch = -1
    latest_file_info = None
    latest_mtime = -1  # 用于记录最新文件的修改时间

    # vvvvvv 正则表达式现在只匹配 .pth 文件 vvvvvv
    pattern = re.compile(r"model_(\d+)_(\d+)x(\d+)_(\d+)c\.pth")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            full_path = os.path.join(path, f)
            mtime = os.path.getmtime(full_path)

            # 如果轮次更大，或者轮次相同但文件是更新的，则更新为最新模型
            if epoch > max_epoch or (epoch == max_epoch and mtime > latest_mtime):
                max_epoch = epoch
                latest_mtime = mtime
                latest_file_info = {
                    'path': f,
                    'epoch': epoch,
                    'res_blocks': int(match.group(2)),
                    'hidden_units': int(match.group(3)),
                    'channels': int(match.group(4))
                }
    return latest_file_info

# ====================== Coach 类 (已更新 learn 方法) ======================


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
        # ==================== 新增诊断日志 开始 ====================
        # 为了避免刷屏，我们只检查批次中的第一个样本
        if batch:
            print("\n[DEBUG Coach] Inspecting a sample from the training batch:")
            _sample_state, sample_policy, sample_value = batch[0]

            # 检查策略向量中是否有非零值
            # np.any() 会在找到任何一个非零元素时返回 True
            if np.any(sample_policy):
                print(f"  - Policy looks OK. Max probability: {np.max(sample_policy):.4f}")
            else:
                # 如果策略向量所有值都是0，这是一个非常危险的信号
                print("  - CRITICAL WARNING: Policy vector is all zeros!")

            print(f"  - Value: {sample_value:.4f}")
        # ==================== 新增诊断日志 结束 ====================
        # ==================== 数据增强核心逻辑 ====================
        augmented_batch = []
        for state, policy, value in batch:
            state_reshaped = np.array(state).reshape(self.args['num_channels'], self.args['board_size'],
                                                     self.args['board_size'])
            policy_reshaped = np.array(policy).reshape(self.args['board_size'], self.args['board_size'])

            # 循环应用8种对称变换
            for i in range(1, 5):  # 旋转 0, 90, 180, 270 度
                # 旋转
                aug_state_rot = np.rot90(state_reshaped, i, axes=(1, 2))
                aug_policy_rot = np.rot90(policy_reshaped, i)
                augmented_batch.append((aug_state_rot.flatten(), aug_policy_rot.flatten(), value))

                # 旋转后再水平翻转
                aug_state_flipped = np.flip(aug_state_rot, axis=2)
                aug_policy_flipped = np.flip(aug_policy_rot, axis=1)
                augmented_batch.append((aug_state_flipped.flatten(), aug_policy_flipped.flatten(), value))
        # ========================================================
        # 使用增强后的数据进行训练
        states, target_policies, target_values = zip(*augmented_batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        target_policies = torch.tensor(np.array(target_policies), dtype=torch.float32).to(device)
        target_values = torch.tensor(np.array(target_values), dtype=torch.float32).unsqueeze(1).to(device)
        states = states.view(-1, self.args['num_channels'], self.args['board_size'], self.args['board_size'])
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            pred_log_policies, pred_values = self.model(states)
            policy_loss = -torch.sum(target_policies * pred_log_policies) / len(target_policies)
            value_loss = F.mse_loss(pred_values, target_values)
            total_loss = policy_loss + self.args['value_loss_weight'] * value_loss
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 请用这个新版本替换旧的 learn 函数

    def learn(self, start_epoch=1):
        for i in range(start_epoch, start_epoch + self.args['num_iterations']):
            print(f"------ 迭代轮次: {i} ------")

            print("步骤1：启动纯C++引擎进行自我对弈 (此过程将阻塞)...")
            final_data_queue = queue.Queue()

            cpp_args = {k: v for k, v in self.args.items()}

            # ==================== 新的、更健壮的模型查找逻辑 开始 ====================
            model_to_use_epoch = i - 1

            # 1. 直接根据当前配置构建期望的、精确的模型文件名
            expected_res_blocks = self.args['num_res_blocks']
            expected_hidden = self.args['num_hidden']
            expected_channels = self.args['num_channels']  # <--- 获取通道数
            model_to_use_path_pt = f"model_{model_to_use_epoch}_{expected_res_blocks}x{expected_hidden}_{expected_channels}c.pt"

            # 2. 检查这个精确的文件是否存在
            if not os.path.exists(model_to_use_path_pt):
                # 如果不存在，打印警告并回退到旧的模糊搜索逻辑，以确保最大兼容性
                print(f"警告：无法找到与当前配置完全匹配的模型 '{model_to_use_path_pt}'。")
                print("将回退到模糊搜索模式...")

                model_to_use_path_pt = None  # 重置路径
                pattern = re.compile(f"model_{model_to_use_epoch}_.*\\.pt")
                # 寻找最新的一个模型
                latest_found_time = -1
                for f in os.listdir("."):
                    if pattern.match(f):
                        file_time = os.path.getmtime(f)
                        if file_time > latest_found_time:
                            latest_found_time = file_time
                            model_to_use_path_pt = f

                if model_to_use_path_pt:
                    print(f"找到最近修改的后备模型: '{model_to_use_path_pt}'。注意：这可能与C++数据格式不匹配！")

            # 3. 如果以上两种方法都找不到，再尝试最原始的文件名格式
            if model_to_use_path_pt is None:
                simple_path = f"model_{model_to_use_epoch}.pt"
                if os.path.exists(simple_path):
                    model_to_use_path_pt = simple_path
                else:
                    print(f"【严重错误】无法找到第 {model_to_use_epoch} 轮的任何.pt模型文件！程序退出。")
                    return
            # ==================== 新的、更健壮的模型查找逻辑 结束 ====================

            print(f"[Python Coach] 指示C++引擎使用模型: {model_to_use_path_pt}")
            cpp_mcts_engine.run_parallel_self_play(
                model_to_use_path_pt,
                device.type == 'cuda',
                final_data_queue,
                cpp_args
            )

            print("\n自我对弈完成！正在进行精细化数据收集与筛选...")
            games_processed = 0
            good_steps_collected = 0
            bad_steps_discarded = 0
            with tqdm.tqdm(total=self.args['num_selfPlay_episodes'], desc="处理对局数据") as pbar:
                while games_processed < self.args['num_selfPlay_episodes']:
                    try:
                        result = final_data_queue.get(timeout=2.0)
                        games_processed += 1
                        pbar.update(1)
                        if result.get("type") == "data":
                            game_data = result.get("data", [])
                            good_steps_from_this_game = []
                            # in learn() function
                            # 从 self.args 获取筛选标志，如果config里没写，默认为 True (保持原行为)
                            enable_filtering = self.args.get('filter_zero_policy_data', True)

                            for state, policy, value in game_data:
                                # 如果关闭了筛选，或者策略向量本身是有效的，则保留数据
                                if not enable_filtering or np.any(policy):
                                    good_steps_from_this_game.append((state, policy, value))
                                else:
                                    bad_steps_discarded += 1
                            if good_steps_from_this_game:
                                self.training_data.extend(good_steps_from_this_game)
                                good_steps_collected += len(good_steps_from_this_game)
                    except queue.Empty:
                        print(f"\n警告：数据队列已空，但只处理了 {games_processed} 局。")
                        break

            print(f"\n数据处理完成！")
            print(f"  本轮共收集到 {good_steps_collected} 个有效训练步骤。")
            print(f"  共丢弃了 {bad_steps_discarded} 个无效步骤。")
            print(f"  当前总经验库大小: {len(self.training_data)}")

            print("\n步骤2：训练神经网络...")
            if len(self.training_data) < self.args['batch_size']:
                print("警告：有效数据不足，跳过本轮训练。将使用旧模型进行下一轮自我对弈。")
                # 保存一份旧模型，但轮次+1，以确保下一轮能找到正确的模型文件
                save_model(self.model, i, self.args)
            else:
                self.model.train()
                for _ in tqdm.tqdm(range(self.args['num_epochs']), desc="训练模型"):
                    self.train()
                self.scheduler.step()
                save_model(self.model, i, self.args)

            clear_windows_memory()
        print(f"\n全部训练迭代完成！")

    def evaluate_models(self, model1_info, model2_info):
        print(f"\n------ 开始分组诊断式评估 (C++ 引擎驱动) ------")
        if not model1_info or not model2_info:
            print("评估缺少必要的模型文件，跳过评估。")
            return

        model1_pt_path = model1_info['path'].replace('.pth', '.pt')
        model2_pt_path = model2_info['path'].replace('.pth', '.pt')

        if not os.path.exists(model1_pt_path) or not os.path.exists(model2_pt_path):
            print("评估缺少必要的.pt模型文件，跳过评估。")
            return

        print(f"评估模型 (旧): {model1_pt_path}")
        print(f"评估模型 (新): {model2_pt_path}")

        use_gpu = (device.type == 'cuda')
        total_games = self.args.get('num_eval_games', 100)
        games_per_side = total_games // 2
        if games_per_side == 0:
            print("评估局数过少，无法进行分组评估。")
            return

        eval_args = {
            # 评估专用参数
            'num_eval_games': games_per_side,
            'num_eval_simulations': self.args['num_searches'],  # 使用 num_searches 的值
            'num_cpu_threads': self.args.get('num_cpu_threads', 18),
            'C': self.args['C'],

            # C++引擎初始化Gomoku和模型所需的通用参数
            'board_size': self.args['board_size'],
            'num_rounds': self.args['num_rounds'],
            'history_steps': self.args['history_steps'],
            'num_channels': self.args['num_channels']  # C++端也需要通道数
        }

        # --- 实验一：新模型执先手 (Model 2) ---
        print(f"\n[实验一] 新模型执黑，进行 {games_per_side} 局...")
        results1 = cpp_mcts_engine.run_parallel_evaluation(
            model1_pt_path, model2_pt_path, use_gpu, eval_args, mode=2
        )
        new_as_p1_wins = results1.get("model2_wins", 0)
        old_as_p2_wins = results1.get("model1_wins", 0)
        draws1 = results1.get("draws", 0)

        # --- 实验二：旧模型执先手 (Model 1) ---
        print(f"\n[实验二] 旧模型执黑，进行 {games_per_side} 局...")
        results2 = cpp_mcts_engine.run_parallel_evaluation(
            model1_pt_path, model2_pt_path, use_gpu, eval_args, mode=1
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
        elif overall_win_rate > self.args.get('eval_win_rate', 0.52):
            print("\n【诊断结论】: 新模型有显著提升！👍")
        else:
            print("\n【诊断结论】: 新模型提升不明显或没有提升，可能陷入了局部最优。")


# ==================== 全新的、智能化的主函数逻辑 ====================
if __name__ == '__main__':
    # ------------------ 参数计算中心 ------------------
    history_channels = (args.get('history_steps', 0) + 1) * 4
    meta_channels = 4
    total_channels = history_channels + meta_channels
    args['num_channels'] = total_channels

    print("=" * 50)
    print("MyAIChess 配置加载完成")
    print(f"历史步数: {args.get('history_steps', 0)}")
    print(f"计算出的总输入通道数: {args['num_channels']}")
    print("=" * 50)
    # ----------------------------------------------------

    print(f"将要使用的设备 (主进程/训练): {device}")

    latest_model_info = find_latest_model_file()
    start_epoch = 1

    # 使用动态通道数创建模型实例
    current_model = ExtendedConnectNet(
        board_size=args['board_size'],
        num_res_blocks=args['num_res_blocks'],
        num_hidden=args['num_hidden'],
        num_channels=args['num_channels']
    ).to(device)

    model_info_before_training = None

    if latest_model_info is None:
        print("未找到任何已有模型，将从第 1 轮开始全新训练。")
        start_epoch = 1
        print("正在创建并保存初始随机模型 (model_0)...")
        save_model(current_model, 0, args)
        model_info_before_training = find_latest_model_file()

    else:
        print(f"找到最新模型: {latest_model_info['path']} (第 {latest_model_info['epoch']} 轮)")
        start_epoch = latest_model_info['epoch'] + 1
        model_info_before_training = latest_model_info

        config_blocks = args['num_res_blocks']
        config_hidden = args['num_hidden']
        config_channels = args['num_channels']

        # 检查架构时，同时检查通道数
        is_same_architecture = (latest_model_info['res_blocks'] == config_blocks and
                                latest_model_info['hidden_units'] == config_hidden and
                                latest_model_info['channels'] == config_channels)

        if is_same_architecture:
            print("模型结构与当前配置一致，直接加载权重继续训练。")
            try:
                current_model.load_state_dict(torch.load(latest_model_info['path'], map_location=device))
                print("权重加载成功！")
            except Exception as e:
                print(f"加载权重失败: {e}，将从随机权重开始。")
                start_epoch = 1
        else:
            print("模型结构与当前配置不一致，将执行自动迁移学习。")
            print(
                f"  旧结构: {latest_model_info['res_blocks']} res_blocks, {latest_model_info['hidden_units']} hidden, {latest_model_info.get('channels', 'N/A')} channels")
            print(f"  新结构: {config_blocks} res_blocks, {config_hidden} hidden, {config_channels} channels")
            try:
                current_model = transfer_weights(current_model, latest_model_info['path'])

                print("为迁移学习后的新模型创建匹配的 .pt 文件...")
                save_model(current_model, latest_model_info['epoch'], args)

                # vvvvvvvv 【请务必确认增加了此段代码】 vvvvvvvv
                # 在保存了迁移后的新模型后，必须立即重新查找并更新评估基准，
                # 以确保 model_info_before_training 指向的是这个新架构的模型。
                print("...更新评估基准为新架构的模型...")
                model_info_before_training = find_latest_model_file()
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            except Exception as e:
                print(f"迁移学习失败: {e}，将从随机权重开始训练新结构模型。")

    coach = Coach(current_model, args)
    coach.learn(start_epoch=start_epoch)

    model_info_after_training = find_latest_model_file()

    if model_info_before_training and model_info_after_training and \
            model_info_before_training['path'] != model_info_after_training['path']:
        coach.evaluate_models(model_info_before_training, model_info_after_training)
    else:
        print("\n未进行有效的新一轮训练或未找到旧模型，跳过评估。")

    print("\n训练全部完成，正在手动清理内存...")
    if 'coach' in locals() and hasattr(coach, 'training_data'):
        coach.training_data.clear()
        print("经验回放池已清空。")
    import gc

    gc.collect()
    print("内存清理完成。程序即将退出。")
