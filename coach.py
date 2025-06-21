# file: coach.py (集成了每轮评估与滑动窗口经验池的最终版)
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
from collections import deque
import platform
import ctypes

from neural_net import ExtendedConnectNet
from config import args
import cpp_mcts_engine


def get_augmented_data(state, policy, board_size, num_channels):
    """
    对单个训练样本进行8种对称变换的数据增强。
    此版本确保所有增强数据都是独立的内存副本。
    """
    state_np = np.array(state).reshape(num_channels, board_size, board_size)
    policy_np = np.array(policy).reshape(board_size, board_size)
    augmented_data = []
    for i in range(4):
        rotated_state = np.rot90(state_np, i, axes=(1, 2))
        rotated_policy = np.rot90(policy_np, i)
        augmented_data.append((rotated_state.copy(), rotated_policy.flatten()))
        flipped_state = np.flip(rotated_state, axis=2)
        flipped_policy = np.flip(rotated_policy, axis=1)
        augmented_data.append((flipped_state.copy(), flipped_policy.flatten()))
    return augmented_data


def clear_windows_memory():
    if platform.system() == "Windows":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            print("[System] Windows memory working set has been cleared.")
        except Exception as e:
            print(f"[System] Failed to clear Windows memory working set: {e}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transfer_weights(new_model, path_to_old_weights):
    """
    将旧模型（通常是较小的模型）的权重加载到新模型中。
    只加载层名和权重形状都匹配的层。
    """
    print(f"--- 启动迁移学习，从 '{path_to_old_weights}' 加载权重 ---")
    old_state_dict = torch.load(path_to_old_weights, map_location=torch.device('cpu'))
    new_model_state_dict = new_model.state_dict()
    loaded_count, skipped_count = 0, 0
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
    num_channels = args['num_channels']
    base_filename = f"model_{epoch}_{args['num_res_blocks']}x{args['num_hidden']}_{num_channels}c"
    model_path_pth = f"{base_filename}.pth"
    model_path_pt = f"{base_filename}.pt"

    torch.save(model.state_dict(), model_path_pth)
    print(f"模型 {model_path_pth} 已保存。")

    model.eval()
    example_input = torch.rand(1, num_channels, args['board_size'], args['board_size']).to(device)
    try:
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(model_path_pt)
        print(f"TorchScript模型 {model_path_pt} 已成功导出。")
    except Exception as e:
        print(f"【错误】导出TorchScript模型失败: {e}")


def find_latest_model_file():
    """
    查找最新的模型文件，当轮次（epoch）相同时，选择最近被修改的文件。
    """
    path = "."
    max_epoch = -1
    latest_file_info = None
    latest_mtime = -1
    pattern = re.compile(r"model_(\d+)_(\d+)x(\d+)_(\d+)c\.pth")

    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            full_path = os.path.join(path, f)
            mtime = os.path.getmtime(full_path)
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


class Coach:
    def __init__(self, model, args):
        self.model = model  # self.model 将始终代表“当前最优模型”
        self.args = args
        self.training_data = deque(maxlen=self.args['data_max_size'])
        self.scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ====================== 【核心改动 1】train函数参数化 ======================
    # 让train函数可以灵活地训练任何模型，而不仅仅是self.model
    # 这是修改后的 train 函数
    def train(self, model_to_train, optimizer, scheduler=None):
        """
        对给定的模型和优化器执行一个训练周期，并返回平均损失。
        """
        model_to_train.train()

        # --- 新增：初始化损失列表 ---
        policy_losses = []
        value_losses = []

        for _ in tqdm.tqdm(range(self.args.get('training_steps_per_iteration', 500)), desc="训练模型 Steps"):
            if len(self.training_data) < self.args['batch_size']:
                # ...
                continue

            batch = random.sample(self.training_data, self.args['batch_size'])
            # ... (数据增强逻辑不变) ...
            augmented_batch = []
            for state, policy, value in batch:
                augmented_samples = get_augmented_data(state, policy, self.args['board_size'],
                                                       self.args['num_channels'])
                for aug_s, aug_p in augmented_samples:
                    augmented_batch.append((aug_s, aug_p, value))

            states, target_policies, target_values = zip(*augmented_batch)
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            target_policies = torch.tensor(np.array(target_policies), dtype=torch.float32).to(device)
            target_values = torch.tensor(np.array(target_values), dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred_log_policies, pred_values = model_to_train(states)
                policy_loss = -torch.sum(target_policies * pred_log_policies) / len(target_policies)
                value_loss = F.mse_loss(pred_values, target_values)
                total_loss = policy_loss + self.args['value_loss_weight'] * value_loss

            # --- 新增：将当前批次的损失记录下来 ---
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

            self.scaler.scale(total_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            if scheduler is not None:
                scheduler.step()

        # --- 新增：计算并返回平均损失 ---
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0
        avg_value_loss = np.mean(value_losses) if value_losses else 0
        return avg_policy_loss, avg_value_loss

    # file: coach.py
    # 请用这个新版本的函数，完整替换掉文件中旧的 learn 函数

    # file: coach.py
    # 这是最终的、实现了“目标驱动”逻辑的 learn 函数，请用它替换旧版本

    def learn(self):
        """
        执行包含“自对弈->训练->评估->晋升/丢弃”循环的完整学习过程。
        此版本会持续运行，直到成功晋升了 num_iterations 指定的代数。
        """
        best_model_info = find_latest_model_file()
        if not best_model_info:
            print("【严重错误】无法找到任何模型文件！程序退出。")
            return

        # --- 新的目标驱动循环控制逻辑 ---
        model_was_promoted = True  # 标记上一轮尝试是否成功，决定本轮自对弈规模
        current_model_epoch = best_model_info['epoch']  # 当前冠军模型的代数
        elo_best_model = 1500  # Elo每次运行可以重新计算

        # 根据配置计算最终的目标代数
        num_successful_promotions_to_achieve = self.args['num_iterations']
        target_epoch = current_model_epoch + num_successful_promotions_to_achieve

        print(
            f"训练启动：当前模型轮次 {current_model_epoch}，目标轮次 {target_epoch} (需要 {num_successful_promotions_to_achieve} 次成功晋升)")

        attempt_num = 0
        # 循环直到当前模型代数达到目标
        while current_model_epoch < target_epoch:
            attempt_num += 1
            promotions_needed = target_epoch - current_model_epoch
            # 日志现在会显示还需要多少次成功晋升
            print(
                f"\n{'=' * 20} 尝试周期: {attempt_num} | 目标: model_{current_model_epoch + 1} (还需 {promotions_needed} 次晋升) {'=' * 20}")

            # --- 步骤 1: 使用当前最优模型进行自对弈 ---
            best_model_path_pt = best_model_info['path'].replace('.pth', '.pt')
            cpp_args = self.args.copy()
            if model_was_promoted:
                print(f"步骤1: 模型刚晋升或首次运行，执行一轮完整的自对弈 ({cpp_args['num_selfPlay_episodes']} 局)...")
            else:
                small_episodes = max(1, int(cpp_args['num_selfPlay_episodes'] * 0.1))
                cpp_args['num_selfPlay_episodes'] = small_episodes
                print(f"步骤1: 上次尝试未晋升，执行一轮小规模增量自对弈 ({small_episodes} 局)...")

            print(f"   使用模型: '{best_model_path_pt}'")

            final_data_queue = queue.Queue()
            cpp_mcts_engine.run_parallel_self_play(
                best_model_path_pt,
                device.type == 'cuda',
                final_data_queue,
                cpp_args
            )

            # --- 步骤 2: 收集数据并放入“滑动窗口”经验池 ---
            print("自对弈完成！正在收集和筛选新数据...")
            games_processed, good_steps_collected, bad_steps_discarded = 0, 0, 0
            policy_entropies = []
            while not final_data_queue.empty():
                try:
                    result = final_data_queue.get_nowait()
                    if result.get("type") == "data":
                        games_processed += 1
                        game_data = result.get("data", [])
                        enable_filtering = self.args.get('filter_zero_policy_data', True)
                        for state, policy, value in game_data:
                            if not enable_filtering or np.any(policy):
                                self.training_data.append((state, policy, value))
                                p_vec = np.array(policy)
                                p_vec = p_vec[p_vec > 0]
                                if p_vec.size > 0:
                                    entropy = -np.sum(p_vec * np.log2(p_vec))
                                    policy_entropies.append(entropy)
                                good_steps_collected += 1
                            else:
                                bad_steps_discarded += 1
                except queue.Empty:
                    break

            print(
                f"数据处理完成！本轮共处理 {games_processed} 局, 收集到 {good_steps_collected} 个有效步骤, 丢弃 {bad_steps_discarded} 个。")
            print(f"当前总经验库大小: {len(self.training_data)}")

            if len(self.training_data) < self.args['batch_size']:
                print("警告：经验池数据不足，跳过本次训练和评估。")
                model_was_promoted = False
                continue

            promotion_achieved_this_attempt = False

            # --- 步骤 3: 训练与评估候选模型 ---
            for candidate_idx in range(self.args.get('num_candidates_to_train', 1)):
                print(
                    f"\n{'--' * 15} 正在尝试第 {candidate_idx + 1} / {self.args.get('num_candidates_to_train', 1)} 个候选模型 {'--' * 15}")

                print("\n步骤3.1: 训练候选模型...")
                candidate_model = ExtendedConnectNet(
                    board_size=self.args['board_size'], num_res_blocks=self.args['num_res_blocks'],
                    num_hidden=self.args['num_hidden'], num_channels=self.args['num_channels']
                ).to(device)
                candidate_model.load_state_dict(self.model.state_dict())

                optimizer = optim.Adam(candidate_model.parameters(), lr=self.args['learning_rate'], weight_decay=0.0001)
                avg_p_loss, avg_v_loss = self.train(candidate_model, optimizer)
                print(f"  - 训练损失: Policy Loss={avg_p_loss:.4f}, Value Loss={avg_v_loss:.4f}")

                print("\n步骤3.2: 评估候选模型 vs. 最优模型...")
                candidate_model_path_pt = f"candidate_{candidate_idx}.pt"
                candidate_model.eval()
                example_input = torch.rand(1, self.args['num_channels'], self.args['board_size'],
                                           self.args['board_size']).to(device)
                traced_script_module = torch.jit.trace(candidate_model, example_input)
                traced_script_module.save(candidate_model_path_pt)

                use_gpu = (device.type == 'cuda')
                games_per_side = self.args.get('num_eval_games', 20) // 2
                eval_args = self.args.copy()
                eval_args['num_eval_games'] = games_per_side
                eval_args['num_eval_simulations'] = self.args['num_searches']

                results1 = cpp_mcts_engine.run_parallel_evaluation(
                    best_model_path_pt, candidate_model_path_pt, use_gpu, eval_args, mode=2
                )
                results2 = cpp_mcts_engine.run_parallel_evaluation(
                    best_model_path_pt, candidate_model_path_pt, use_gpu, eval_args, mode=1
                )

                total_new_model_wins = results1.get("model2_wins", 0) + results2.get("model2_wins", 0)
                total_old_model_wins = results1.get("model1_wins", 0) + results2.get("model1_wins", 0)
                total_draws = results1.get("draws", 0) + results2.get("draws", 0)
                total_games = total_new_model_wins + total_old_model_wins + total_draws
                win_rate = total_new_model_wins / total_games if total_games > 0 else 0

                print("\n评估总结:")
                print(
                    f"  - 新模型执黑时 (新 vs 旧): {results1.get('model2_wins', 0)} 胜 / {results1.get('model1_wins', 0)} 负 / {results1.get('draws', 0)} 平")
                print(
                    f"  - 新模型执白时 (旧 vs 新): {results2.get('model2_wins', 0)} 胜 / {results2.get('model1_wins', 0)} 负 / {results2.get('draws', 0)} 平")
                print(
                    f"  - 综合战绩 (新 vs 旧): {total_new_model_wins} 胜 / {total_old_model_wins} 负 / {total_draws} 平")
                print(f"  - 新模型综合胜率: {win_rate:.2%}")

                elo_candidate = elo_best_model
                expected_win_rate_candidate = 1 / (1 + 10 ** ((elo_best_model - elo_candidate) / 400))
                actual_score_candidate = total_new_model_wins + 0.5 * total_draws
                expected_score_candidate = expected_win_rate_candidate * total_games
                k_factor = self.args.get('elo_k_factor', 32)
                new_elo_candidate = elo_candidate + k_factor * (actual_score_candidate - expected_score_candidate)

                print(f"  - Elo 评级: BestNet ({elo_best_model:.0f}) vs Candidate ({elo_best_model:.0f})")
                print(
                    f"  - Elo 变化: Candidate Elo -> {new_elo_candidate:.0f} ({new_elo_candidate - elo_candidate:+.0f})")

                if os.path.exists(candidate_model_path_pt):
                    os.remove(candidate_model_path_pt)

                if win_rate >= 0.55:
                    next_model_epoch = current_model_epoch + 1

                    print(
                        f"【模型晋升】候选 {candidate_idx + 1} 胜率达标，将其保存为 model_{next_model_epoch} 并设为新的最优模型。")
                    elo_best_model = new_elo_candidate
                    self.model.load_state_dict(candidate_model.state_dict())
                    save_model(self.model, next_model_epoch, self.args)
                    best_model_info = find_latest_model_file()

                    # 更新循环控制变量，向目标迈进一步
                    current_model_epoch = next_model_epoch
                    promotion_achieved_this_attempt = True

                    break
                else:
                    print(f"【模型丢弃】候选 {candidate_idx + 1} 胜率未达标。")

            model_was_promoted = promotion_achieved_this_attempt

            if not model_was_promoted:
                print(f"\n--- 本次尝试未能晋升，将继续尝试击败 model_{current_model_epoch} ---")

            print(f"\n{'=' * 20} 尝试周期 {attempt_num} 总结 {'=' * 20}")
            print("性能指标:")
            print(f"  - 当前最优模型 Elo: {elo_best_model:.0f} (model_{current_model_epoch})")
            print("行为统计 (来自本轮自对弈):")
            avg_entropy = np.mean(policy_entropies) if policy_entropies else 0
            print(f"  - 平均MCTS策略熵: {avg_entropy:.3f} bits")
            print(f"{'=' * 56}")

            clear_windows_memory()

        print(
            f"\n训练目标达成！已成功晋升 {num_successful_promotions_to_achieve} 代新模型，最终模型为 model_{current_model_epoch}。")

    # 这个函数保留，用于最终的、更详细的评估报告，或者可以被手动调用
    def evaluate_models(self, model1_info, model2_info):
        # ... 此函数内容保持不变 ...
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
            'num_eval_games': games_per_side,
            'num_eval_simulations': self.args['num_searches'],
            'num_cpu_threads': self.args.get('num_cpu_threads', 18),
            'C': self.args['C'],
            'mcts_batch_size': self.args['mcts_batch_size'],
            'board_size': self.args['board_size'],
            'num_rounds': self.args['num_rounds'],
            'history_steps': self.args['history_steps'],
            'num_channels': self.args['num_channels'],
            'enable_territory_heuristic': self.args.get('enable_territory_heuristic', False),
            'territory_heuristic_weight': self.args.get('territory_heuristic_weight', 0.0),
            'enable_territory_penalty': self.args.get('enable_territory_penalty', False),
            'territory_penalty_strength': self.args.get('territory_penalty_strength', 0.0),
            'enable_ineffective_connection_penalty': self.args.get('enable_ineffective_connection_penalty', False),
            'ineffective_connection_penalty_factor': self.args.get('ineffective_connection_penalty_factor', 0.1),
        }

        print(f"\n[实验一] 新模型执黑，进行 {games_per_side} 局...")
        results1 = cpp_mcts_engine.run_parallel_evaluation(
            model1_pt_path, model2_pt_path, use_gpu, eval_args, mode=2
        )
        new_as_p1_wins = results1.get("model2_wins", 0)
        old_as_p2_wins = results1.get("model1_wins", 0)
        draws1 = results1.get("draws", 0)

        print(f"\n[实验二] 旧模型执黑，进行 {games_per_side} 局...")
        results2 = cpp_mcts_engine.run_parallel_evaluation(
            model1_pt_path, model2_pt_path, use_gpu, eval_args, mode=1
        )
        old_as_p1_wins = results2.get("model1_wins", 0)
        new_as_p2_wins = results2.get("model2_wins", 0)
        draws2 = results2.get("draws", 0)

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
            print("\n【诊断结论】: 新模型提升不明显或没有提升。")


if __name__ == '__main__':
    # ====================== 主函数逻辑保持不变 ======================
    history_channels = (args.get('history_steps', 0) + 1) * 4
    meta_channels = 4
    total_channels = history_channels + meta_channels
    args['num_channels'] = total_channels

    print("=" * 50)
    print("MyAIChess 配置加载完成")
    print(f"历史步数: {args.get('history_steps', 0)}")
    print(f"计算出的总输入通道数: {args['num_channels']}")
    print("=" * 50)

    print(f"将要使用的设备 (主进程/训练): {device}")

    latest_model_info = find_latest_model_file()
    start_epoch = 1

    current_model = ExtendedConnectNet(
        board_size=args['board_size'],
        num_res_blocks=args['num_res_blocks'],
        num_hidden=args['num_hidden'],
        num_channels=args['num_channels']
    ).to(device)

    # --- 模型加载和迁移学习逻辑保持不变 ---
    if latest_model_info is None:
        print("未找到任何已有模型，将从第 1 轮开始全新训练。")
        start_epoch = 1
        print("正在创建并保存初始随机模型 (model_0)...")
        save_model(current_model, 0, args)
    else:
        print(f"找到最新模型: {latest_model_info['path']} (第 {latest_model_info['epoch']} 轮)")
        start_epoch = latest_model_info['epoch'] + 1
        config_blocks = args['num_res_blocks']
        config_hidden = args['num_hidden']
        config_channels = args['num_channels']
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
            except Exception as e:
                print(f"迁移学习失败: {e}，将从随机权重开始训练新结构模型。")

    # --- 启动训练 ---
    coach = Coach(current_model, args)
    coach.learn()

    print("\n训练全部完成，正在手动清理内存...")
    if 'coach' in locals() and hasattr(coach, 'training_data'):
        coach.training_data.clear()
        print("经验回放池已清空。")
    import gc

    gc.collect()
    print("内存清理完成。程序即将退出。")
