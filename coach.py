# file: coach.py (已修正 NameError 的最终完整版)
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
from collections import deque, Counter
import platform
import ctypes
import pickle

from neural_net import ExtendedConnectNet
from config import args
import cpp_mcts_engine


def get_augmented_data(state, policy, board_size, num_channels):
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


def get_move_number_from_state(state, args):
    """从状态向量中解析出当前是第几步"""
    try:
        num_channels = args['num_channels']
        board_size = args['board_size']
        max_total_moves = args['num_rounds'] * 2
        history_steps = args.get('history_steps', 0)

        # 游戏进度在元数据通道的第2个平面 (索引为1)
        progress_channel_idx = (history_steps + 1) * 4 + 1

        state_np = np.array(state).reshape(num_channels, board_size, board_size)

        # 从该平面任意位置获取进度值
        progress = state_np[progress_channel_idx, 0, 0]

        # 反归一化得到步数
        move_number = round(progress * max_total_moves)
        return int(move_number)
    except Exception as e:
        print(f"[警告] 从状态解析步数失败: {e}")
        return -1 # 返回一个错误标识


def clear_windows_memory():
    if platform.system() == "Windows":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            print("[System] Windows memory working set has been cleared.")
        except Exception as e:
            print(f"[System] Failed to clear Windows memory working set: {e}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transfer_weights(new_model, path_to_old_weights):
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


# --- 在 coach.py 中添加此函数 ---
def remove_old_models(args, keep_max=3):
    """
    清理旧模型，只保留最新的 keep_max 个版本的模型文件 (.pth 和 .pt)。
    """
    path = "."
    # 匹配模型文件的正则，例如: model_10_5x128_20c.pth
    # 这里的正则要足够宽容以匹配 .pth 和 .pt
    pattern = re.compile(r"model_(\d+)_.*")

    # 1. 扫描所有模型文件并按 epoch 分组
    found_models = {}  # 格式: {epoch: [file_path1, file_path2]}

    for f in os.listdir(path):
        # 只处理文件
        if not os.path.isfile(os.path.join(path, f)):
            continue

        match = pattern.match(f)
        if match:
            # 排除 candidate 模型，只处理正式的 model_
            if "candidate" in f:
                continue

            epoch = int(match.group(1))
            if epoch not in found_models:
                found_models[epoch] = []
            found_models[epoch].append(f)

    # 2. 如果总版本数少于保留数，不需要清理
    if len(found_models) <= keep_max:
        return

    # 3. 按 epoch 从小到大排序
    sorted_epochs = sorted(found_models.keys())

    # 4. 找出需要删除的 epoch (即除了最后 keep_max 个之外的所有 epoch)
    epochs_to_delete = sorted_epochs[:-keep_max]

    # 5. 执行删除
    print(f"\n[清理] 正在移除旧模型权重 (保留最近 {keep_max} 个版本)...")
    for epoch in epochs_to_delete:
        files = found_models[epoch]
        for file_name in files:
            try:
                os.remove(file_name)
                print(f"  - 已删除: {file_name}")
            except Exception as e:
                print(f"  - 删除失败 {file_name}: {e}")


def save_model(model, epoch, args):
    num_channels = args['num_channels']
    base_filename = f"model_{epoch}_{args['num_res_blocks']}x{args['num_hidden']}_{num_channels}c"
    model_path_pth = f"{base_filename}.pth"
    model_path_pt = f"{base_filename}.pt"
    torch.save(model.state_dict(), model_path_pth)
    print(f"模型 {model_path_pth} 已保存。")
    model.eval()
    try:
        traced_script_module = torch.jit.trace(model,
                                               torch.rand(1, num_channels, args['board_size'], args['board_size']).to(
                                                   device))
        traced_script_module.save(model_path_pt)
        print(f"TorchScript模型 {model_path_pt} 已成功导出。")
    except Exception as e:
        print(f"【错误】导出TorchScript模型失败: {e}")

    # ==================== 新增代码 ====================
    # 3. 调用清理函数，移除过期的旧权重
    remove_old_models(args, keep_max=3)
    # ================================================


def find_latest_model_file():
    path, max_epoch, latest_file_info, latest_mtime = ".", -1, None, -1
    pattern = re.compile(r"model_(\d+)_(\d+)x(\d+)_(\d+)c\.pth")
    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            epoch, mtime = int(match.group(1)), os.path.getmtime(os.path.join(path, f))
            if epoch > max_epoch or (epoch == max_epoch and mtime > latest_mtime):
                max_epoch, latest_mtime = epoch, mtime
                latest_file_info = {'path': f, 'epoch': epoch, 'res_blocks': int(match.group(2)),
                                    'hidden_units': int(match.group(3)), 'channels': int(match.group(4))}
    return latest_file_info


class Coach:
    def __init__(self, model, args):
        self.model, self.args = model, args
        self.scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.self_play_data = deque(maxlen=self.args['data_max_size'])
        self.expert_data = deque(maxlen=self.args.get('expert_data_max_size', 50000))

    def train(self, model_to_train, optimizer, scheduler=None):
        model_to_train.train()
        policy_losses, value_losses = [], []
        for _ in tqdm.tqdm(range(self.args.get('training_steps_per_iteration', 500)), desc="训练模型 Steps"):
            expert_samples, self_play_samples = [], []
            total_batch_size, expert_ratio = self.args['batch_size'], self.args.get('expert_data_ratio', 0.25)
            expert_batch_size = int(total_batch_size * expert_ratio) if len(self.expert_data) > 0 else 0
            expert_batch_size = min(len(self.expert_data), expert_batch_size)
            if expert_batch_size > 0: expert_samples = random.sample(self.expert_data, expert_batch_size)
            self_play_batch_size = total_batch_size - expert_batch_size
            if len(self.self_play_data) < self_play_batch_size: continue
            self_play_samples = random.sample(self.self_play_data, self_play_batch_size)
            batch = expert_samples + self_play_samples
            if not batch: continue
            augmented_batch = []
            for item in batch:
                # 根据item长度判断是专家数据还是自对弈数据
                if len(item) == 4:
                    state, policy, value, _ = item  # 解包4个元素，忽略最后一个
                else:
                    state, policy, value = item  # 正常解包3个元素

                # 后续的数据增强逻辑保持不变
                if np.any(policy):
                    for aug_s, aug_p in get_augmented_data(state, policy, self.args['board_size'],
                                                           self.args['num_channels']):
                        augmented_batch.append((aug_s, aug_p, value))
                else:
                    augmented_batch.append((np.array(state).reshape(self.args['num_channels'], self.args['board_size'],
                                                                    self.args['board_size']), policy, value))
            states, target_policies, target_values = zip(*augmented_batch)
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            target_policies = torch.tensor(np.array(target_policies), dtype=torch.float32).to(device)
            target_values = torch.tensor(np.array(target_values), dtype=torch.float32).unsqueeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred_log_policies, pred_values = model_to_train(states)
                policy_mask = torch.any(target_policies, dim=1)
                policy_loss = -torch.sum(
                    target_policies[policy_mask] * pred_log_policies[policy_mask]) / policy_mask.sum() if torch.any(
                    policy_mask) else torch.tensor(0.0).to(device)
                value_loss = F.mse_loss(pred_values, target_values)
                total_loss = policy_loss + self.args['value_loss_weight'] * value_loss
            policy_losses.append(policy_loss.item());
            value_losses.append(value_loss.item())
            self.scaler.scale(total_loss).backward();
            self.scaler.step(optimizer);
            self.scaler.update()
            if scheduler: scheduler.step()
        return (np.mean(policy_losses) if policy_losses else 0), (np.mean(value_losses) if value_losses else 0)

    def learn(self):
        self_play_data_file = 'self_play_data.pkl'
        if os.path.exists(self_play_data_file):
            try:
                with open(self_play_data_file, 'rb') as f:
                    self.self_play_data.extend(pickle.load(f))
                print(f"[数据加载] 成功从 '{self_play_data_file}' 加载了 {len(self.self_play_data)} 条自对弈数据。")
            except Exception as e:
                print(f"[警告] 加载 '{self_play_data_file}' 失败: {e}")
        human_data_file = 'human_games.pkl'
        if os.path.exists(human_data_file):
            try:
                with open(human_data_file, 'rb') as f:
                    expert_data_loaded = pickle.load(f)
                self.expert_data.clear();
                self.expert_data.extend(expert_data_loaded)
                print(f"[数据加载] 成功从 '{human_data_file}' 加载了 {len(self.expert_data)} 条专家数据。")
            except Exception as e:
                print(f"[警告] 加载 '{human_data_file}' 失败: {e}")
        print(f"\n--- 启动时总数据: {len(self.self_play_data)} (自对弈) + {len(self.expert_data)} (专家) ---\n")

        best_model_info = find_latest_model_file()
        current_model_epoch = best_model_info['epoch']

        # --- 核心修正：恢复此变量的定义 ---
        num_successful_promotions_to_achieve = self.args['num_iterations']
        target_epoch = current_model_epoch + num_successful_promotions_to_achieve

        print(
            f"训练启动：当前模型轮次 {current_model_epoch}，目标轮次 {target_epoch} (需要 {num_successful_promotions_to_achieve} 次成功晋升)")

        attempt_num, elo_best_model, model_was_promoted = 0, 1500, True
        while current_model_epoch < target_epoch:
            attempt_num += 1
            promotions_needed = target_epoch - current_model_epoch
            print(
                f"\n{'=' * 20} 尝试周期: {attempt_num} | 目标: model_{current_model_epoch + 1} (还需 {promotions_needed} 次晋升) {'=' * 20}")
            best_model_info = find_latest_model_file()
            best_model_path_pt = best_model_info['path'].replace('.pth', '.pt')
            cpp_args = self.args.copy()
            if model_was_promoted:
                print(f"步骤1: 模型刚晋升或首次运行，执行一轮完整的自对弈 ({cpp_args['num_selfPlay_episodes']} 局)...")
            else:
                small_episodes = max(1, int(
                    cpp_args['num_selfPlay_episodes'] * self.args.get('failed_selfplay_ratio', 0.1)))
                cpp_args['num_selfPlay_episodes'] = small_episodes
                print(f"步骤1: 上次尝试未晋升，执行一轮小规模增量自对弈 ({small_episodes} 局)...")
            print(f"   使用模型: '{best_model_path_pt}'")
            final_data_queue = queue.Queue()
            cpp_mcts_engine.run_parallel_self_play(best_model_path_pt, device.type == 'cuda', final_data_queue,
                                                   cpp_args)

            games_processed, good_steps_collected, bad_steps_discarded, policy_entropies = 0, 0, 0, []
            enable_filtering = self.args.get('filter_zero_policy_data', True)
            discarded_samples_for_debug = []
            while not final_data_queue.empty():
                try:
                    result = final_data_queue.get_nowait()
                    if result.get("type") == "data":
                        games_processed += 1
                        for state, policy, value in result.get("data", []):
                            if not enable_filtering or np.any(policy):
                                self.self_play_data.append((state, policy, value));
                                good_steps_collected += 1
                                p_vec = np.array(policy);
                                p_vec = p_vec[p_vec > 0]
                                if p_vec.size > 0: policy_entropies.append(-np.sum(p_vec * np.log2(p_vec)))
                            else:
                                bad_steps_discarded += 1
                                discarded_samples_for_debug.append(state)
                except queue.Empty:
                    break

            print(
                f"数据处理完成！本轮共处理 {games_processed} 局, 收集到 {good_steps_collected} 个有效步骤, 丢弃 {bad_steps_discarded} 个。")
            if bad_steps_discarded > 0 and discarded_samples_for_debug:
                # 解析所有被丢弃数据的步数
                all_move_numbers = [get_move_number_from_state(s, self.args) for s in discarded_samples_for_debug]
                # 使用Counter统计每个步数出现的频率
                move_counts = Counter(all_move_numbers)
                # 按步数排序后输出
                sorted_counts = sorted(move_counts.items())

                print(f"  - [调试信息] 被丢弃数据的步数频率统计 (步数: 次数): {sorted_counts}")
            print(f"当前自对弈池大小: {len(self.self_play_data)}, 专家池大小: {len(self.expert_data)}")
            if len(self.self_play_data) + len(self.expert_data) < self.args['batch_size']:
                print("警告：总数据不足，跳过本次训练。");
                model_was_promoted = False;
                continue

            promotion_achieved = False
            for cand_idx in range(self.args.get('num_candidates_to_train', 1)):
                print(
                    f"\n{'--' * 15} 正在尝试第 {cand_idx + 1} / {self.args.get('num_candidates_to_train', 1)} 个候选模型 {'--' * 15}")
                print("\n步骤3.1: 训练候选模型...")
                cand_model = ExtendedConnectNet(board_size=self.args['board_size'],
                                                num_res_blocks=self.args['num_res_blocks'],
                                                num_hidden=self.args['num_hidden'],
                                                num_channels=self.args['num_channels']).to(device)
                cand_model.load_state_dict(self.model.state_dict())
                optimizer = optim.Adam(cand_model.parameters(), lr=self.args['learning_rate'],
                                       weight_decay=self.args.get('weight_decay', 0.0001))
                avg_p_loss, avg_v_loss = self.train(cand_model, optimizer)
                print(f"  - 训练损失: Policy Loss={avg_p_loss:.4f}, Value Loss={avg_v_loss:.4f}")
                print("\n步骤3.2: 评估候选模型 vs. 最优模型...")
                candidate_model_path_pt = f"candidate_{cand_idx}.pt"
                cand_model.eval();
                traced_script_module = torch.jit.trace(cand_model,
                                                       torch.rand(1, self.args['num_channels'], self.args['board_size'],
                                                                  self.args['board_size']).to(device));
                traced_script_module.save(candidate_model_path_pt)
                games_per_side = self.args.get('num_eval_games', 20) // 2
                eval_args = self.args.copy();
                eval_args['num_eval_games'] = games_per_side;
                eval_args['num_eval_simulations'] = self.args['num_searches']
                res1 = cpp_mcts_engine.run_parallel_evaluation(best_model_path_pt, candidate_model_path_pt,
                                                               device.type == 'cuda', eval_args, mode=2)
                res2 = cpp_mcts_engine.run_parallel_evaluation(best_model_path_pt, candidate_model_path_pt,
                                                               device.type == 'cuda', eval_args, mode=1)
                total_new_wins = res1.get("model2_wins", 0) + res2.get("model2_wins", 0);
                total_old_wins = res1.get("model1_wins", 0) + res2.get("model1_wins", 0);
                total_draws = res1.get("draws", 0) + res2.get("draws", 0)
                total_games = total_new_wins + total_old_wins + total_draws
                win_rate = total_new_wins / total_games if total_games > 0 else 0
                print("\n评估总结:")
                print(
                    f"  - 新模型执黑时 (新 vs 旧): {res1.get('model2_wins', 0)} 胜 / {res1.get('model1_wins', 0)} 负 / {res1.get('draws', 0)} 平")
                print(
                    f"  - 新模型执白时 (旧 vs 新): {res2.get('model2_wins', 0)} 胜 / {res2.get('model1_wins', 0)} 负 / {res2.get('draws', 0)} 平")
                print(f"  - 综合战绩 (新 vs 旧): {total_new_wins} 胜 / {total_old_wins} 负 / {total_draws} 平")
                print(f"  - 新模型综合胜率: {win_rate:.2%}")
                expected_win_rate_candidate = 1 / (1 + 10 ** ((elo_best_model - elo_best_model) / 400));
                actual_score_candidate = total_new_wins + 0.5 * total_draws;
                expected_score_candidate = expected_win_rate_candidate * total_games
                new_elo_candidate = elo_best_model + self.args.get('elo_k_factor', 32) * (
                            actual_score_candidate - expected_score_candidate)
                print(f"  - Elo 评级: BestNet ({elo_best_model:.0f}) vs Candidate ({elo_best_model:.0f})");
                print(
                    f"  - Elo 变化: Candidate Elo -> {new_elo_candidate:.0f} ({new_elo_candidate - elo_best_model:+.0f})")
                if os.path.exists(candidate_model_path_pt): os.remove(candidate_model_path_pt)
                if win_rate >= self.args.get('promotion_win_rate', 0.55):
                    next_model_epoch = current_model_epoch + 1
                    print(
                        f"【模型晋升】候选 {cand_idx + 1} 胜率达标，将其保存为 model_{next_model_epoch} 并设为新的最优模型。")
                    elo_best_model = new_elo_candidate;
                    self.model.load_state_dict(cand_model.state_dict());
                    save_model(self.model, next_model_epoch, self.args)
                    current_model_epoch = next_model_epoch;
                    promotion_achieved = True;
                    break
                else:
                    print(f"【模型丢弃】候选 {cand_idx + 1} 胜率未达标。")
            model_was_promoted = promotion_achieved
            if not model_was_promoted: print(f"\n--- 本次尝试未能晋升，将继续尝试击败 model_{current_model_epoch} ---")
            print(f"\n{'=' * 20} 尝试周期 {attempt_num} 总结 {'=' * 20}")
            print("性能指标:");
            print(f"  - 当前最优模型 Elo: {elo_best_model:.0f} (model_{current_model_epoch})")
            print("行为统计 (来自本轮自对弈):");
            avg_entropy = np.mean(policy_entropies) if policy_entropies else 0
            print(f"  - 平均MCTS策略熵: {avg_entropy:.3f} bits");
            print(f"{'=' * 56}")
            clear_windows_memory()

        print(
            f"\n训练目标达成！已成功晋升 {num_successful_promotions_to_achieve} 代新模型，最终模型为 model_{current_model_epoch}。")

    def evaluate_models(self, model1_info, model2_info):
        print(f"\n------ 开始分组诊断式评估 (C++ 引擎驱动) ------")
        if not model1_info or not model2_info: print("评估缺少必要的模型文件，跳过评估。"); return
        model1_pt_path, model2_pt_path = model1_info['path'].replace('.pth', '.pt'), model2_info['path'].replace('.pth',
                                                                                                                 '.pt')
        if not os.path.exists(model1_pt_path) or not os.path.exists(model2_pt_path): print(
            "评估缺少必要的.pt模型文件，跳过评估。"); return
        print(f"评估模型 (旧): {model1_pt_path}");
        print(f"评估模型 (新): {model2_pt_path}")
        use_gpu, total_games, games_per_side = (device.type == 'cuda'), self.args.get('num_eval_games',
                                                                                      100), self.args.get(
            'num_eval_games', 100) // 2
        if games_per_side == 0: print("评估局数过少，无法进行分组评估。"); return
        eval_args = {k: self.args.get(k) for k in
                     ['num_searches', 'num_cpu_threads', 'C', 'mcts_batch_size', 'board_size', 'num_rounds',
                      'history_steps', 'num_channels', 'enable_territory_heuristic', 'territory_heuristic_weight',
                      'enable_territory_penalty', 'territory_penalty_strength', 'enable_ineffective_connection_penalty',
                      'ineffective_connection_penalty_factor']}
        eval_args['num_eval_games'] = games_per_side;
        eval_args['num_eval_simulations'] = eval_args['num_searches']
        print(f"\n[实验一] 新模型执黑，进行 {games_per_side} 局...")
        results1 = cpp_mcts_engine.run_parallel_evaluation(model1_pt_path, model2_pt_path, use_gpu, eval_args, mode=2)
        new_as_p1_wins, old_as_p2_wins, draws1 = results1.get("model2_wins", 0), results1.get("model1_wins",
                                                                                              0), results1.get("draws",
                                                                                                               0)
        print(f"\n[实验二] 旧模型执黑，进行 {games_per_side} 局...")
        results2 = cpp_mcts_engine.run_parallel_evaluation(model1_pt_path, model2_pt_path, use_gpu, eval_args, mode=1)
        old_as_p1_wins, new_as_p2_wins, draws2 = results2.get("model1_wins", 0), results2.get("model2_wins",
                                                                                              0), results2.get("draws",
                                                                                                               0)
        total_new_wins, total_old_wins, total_draws = new_as_p1_wins + new_as_p2_wins, old_as_p1_wins + old_as_p2_wins, draws1 + draws2
        overall_win_rate = total_new_wins / total_games if total_games > 0 else 0
        print("\n------ 诊断评估结果 ------")
        print(f"新模型执先手时，战绩 (新 vs 旧 | 胜/负/平): {new_as_p1_wins} / {old_as_p2_wins} / {draws1}")
        print(f"旧模型执先手时，战绩 (旧 vs 新 | 胜/负/平): {old_as_p1_wins} / {new_as_p2_wins} / {draws2}")
        print("---------------------------------");
        print(f"综合战绩 - 新 vs 旧 (胜/负/平): {total_new_wins} / {total_old_wins} / {total_draws}");
        print(f"新模型综合胜率: {overall_win_rate:.2%}")
        if games_per_side > 0 and (new_as_p1_wins / games_per_side) > 0.9 and (old_as_p1_wins / games_per_side) > 0.9:
            print("\n【诊断结论】: AI已发现并掌握了 '先手必胜' 策略。")
        elif overall_win_rate > self.args.get('eval_win_rate', 0.52):
            print("\n【诊断结论】: 新模型有显著提升！👍")
        else:
            print("\n【诊断结论】: 新模型提升不明显或没有提升。")


if __name__ == '__main__':
    history_channels = (args.get('history_steps', 0) + 1) * 4
    meta_channels = 4
    total_channels = history_channels + meta_channels
    args['num_channels'] = total_channels
    print("=" * 50);
    print("MyAIChess 配置加载完成");
    print(f"历史步数: {args.get('history_steps', 0)}");
    print(f"计算出的总输入通道数: {args['num_channels']}")
    print("=" * 50);
    print(f"将要使用的设备 (主进程/训练): {device}")
    latest_model_info = find_latest_model_file()
    current_model = ExtendedConnectNet(board_size=args['board_size'], num_res_blocks=args['num_res_blocks'],
                                       num_hidden=args['num_hidden'], num_channels=args['num_channels']).to(device)
    if latest_model_info is None:
        print("未找到任何已有模型，将从第 1 轮开始全新训练。");
        print("正在创建并保存初始随机模型 (model_0)...")
        save_model(current_model, 0, args)
    else:
        print(f"找到最新模型: {latest_model_info['path']} (第 {latest_model_info['epoch']} 轮)")
        config_blocks, config_hidden, config_channels = args['num_res_blocks'], args['num_hidden'], args['num_channels']
        is_same_architecture = (latest_model_info['res_blocks'] == config_blocks and latest_model_info[
            'hidden_units'] == config_hidden and latest_model_info['channels'] == config_channels)
        if is_same_architecture:
            print("模型结构与当前配置一致，直接加载权重继续训练。")
            try:
                current_model.load_state_dict(torch.load(latest_model_info['path'], map_location=device));
                print("权重加载成功！")
            except Exception as e:
                print(f"加载权重失败: {e}，将从随机权重开始。")
        else:
            print("模型结构与当前配置不一致，将执行自动迁移学习。")
            try:
                current_model = transfer_weights(current_model, latest_model_info['path'])
                print("为迁移学习后的新模型创建匹配的 .pt 文件...");
                save_model(current_model, latest_model_info['epoch'], args)
            except Exception as e:
                print(f"迁移学习失败: {e}，将从随机权重开始训练新结构模型。")

    coach = Coach(current_model, args)
    coach.learn()

    print("\n训练全部完成，正在保存自对弈经验池...")
    try:
        with open('self_play_data.pkl', 'wb') as f:
            pickle.dump(coach.self_play_data, f)
        print(f"成功将 {len(coach.self_play_data)} 条自对弈数据保存到 'self_play_data.pkl'。")
    except Exception as e:
        print(f"[错误] 保存自对弈经验池失败: {e}")
    print("\n正在手动清理内存...")
    if 'coach' in locals():
        if hasattr(coach, 'self_play_data'): coach.self_play_data.clear()
        if hasattr(coach, 'expert_data'): coach.expert_data.clear()
    print("内存中的经验池已清空。")
    import gc

    gc.collect()
    clear_windows_memory()
    print("内存清理完成。程序即将退出。")