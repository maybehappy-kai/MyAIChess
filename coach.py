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


def decode_state_to_game_params(state_flat, args):
    """将扁平化的状态向量解码为C++ Gomoku构造函数所需的参数"""
    bs = args['board_size']
    num_channels = args['num_channels']
    state_np = np.array(state_flat, dtype=np.float32).reshape(num_channels, bs, bs)

    # 1. 解析元数据
    hist_steps = args.get('history_steps', 0)
    meta_idx = (hist_steps + 1) * 4

    # 玩家指示器 (Plane 0 of meta): 全1为黑(1), 全0为白(-1)
    player_val = state_np[meta_idx, 0, 0] if meta_idx < num_channels else 1.0
    current_player = 1 if player_val > 0.5 else -1

    # 步数 (Plane 1 of meta)
    # 也可以直接用 get_move_number_from_state，但为了性能这里直接解
    progress = state_np[meta_idx + 1, 0, 0] if (meta_idx + 1) < num_channels else 0.0
    max_moves = args['num_rounds'] * 2
    current_move_number = int(round(progress * max_moves))
    current_move_number = max(0, min(max_moves, current_move_number))

    # 构建 Bitboards (每方2个uint64)
    def make_bitboards(flat_grid):
        b0, b1 = 0, 0
        for i, val in enumerate(flat_grid):
            if val > 0.5:
                if i < 64:
                    b0 |= (1 << i)
                else:
                    b1 |= (1 << (i - 64))
        return [b0, b1]

    def decode_relative_planes(channel_offset):
        p_stones = state_np[channel_offset + 0].flatten()
        o_stones = state_np[channel_offset + 1].flatten()
        p_terr = state_np[channel_offset + 2].flatten()
        o_terr = state_np[channel_offset + 3].flatten()

        p_s_bits = make_bitboards(p_stones)
        o_s_bits = make_bitboards(o_stones)
        p_t_bits = make_bitboards(p_terr)
        o_t_bits = make_bitboards(o_terr)

        if current_player == 1:  # Current is Black
            return {
                "black_stones": p_s_bits,
                "white_stones": o_s_bits,
                "black_territory": p_t_bits,
                "white_territory": o_t_bits,
            }
        # Current is White
        return {
            "black_stones": o_s_bits,
            "white_stones": p_s_bits,
            "black_territory": o_t_bits,
            "white_territory": p_t_bits,
        }

    # 2. 当前状态
    decoded = decode_relative_planes(0)

    # 3. 恢复历史状态 (T-1, T-2, ...)，供评估起始局面复原使用
    history_list = []
    available_history = min(hist_steps, current_move_number)
    for t in range(1, available_history + 1):
        channel_offset = t * 4
        if channel_offset + 3 >= meta_idx:
            break
        history_list.append(decode_relative_planes(channel_offset))

    decoded["current_player"] = current_player
    decoded["current_move_number"] = current_move_number
    decoded["history"] = history_list
    return decoded


def prepare_eval_states(data_pool, num_needed, args):
    """从数据池中筛选合适的开局状态"""
    candidates = []
    # 筛选前10步以内的局面
    for item in data_pool:
        # 兼容不同格式 (state, policy, value) 或 (s, p, v, human_player)
        state = item[0]
        move_num = get_move_number_from_state(state, args)
        if 0 <= move_num < 10:
            candidates.append(state)

    if not candidates:
        print("[警告] 数据池中没有符合条件的开局(步数<10)，将使用空棋盘。")
        # 不能依赖 data_pool[0]，因为 data_pool 可能为空。
        # 直接按配置构造一个空状态，并显式标记黑棋先手。
        empty_state_np = np.zeros((args['num_channels'], args['board_size'], args['board_size']), dtype=np.float32)
        meta_idx = (args.get('history_steps', 0) + 1) * 4
        if meta_idx < args['num_channels']:
            empty_state_np[meta_idx].fill(1.0)
        empty_state = decode_state_to_game_params(empty_state_np.flatten(), args)
        return [empty_state] * num_needed

    # 随机采样并循环填充
    selected = []
    while len(selected) < num_needed:
        choice = random.choice(candidates)
        selected.append(decode_state_to_game_params(choice, args))
    return selected[:num_needed]


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


def extract_model_epoch(filename):
    """从模型文件名中提取 epoch，失败时返回 None。"""
    match = re.search(r"model_(\d+)", os.path.basename(filename))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


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
    tmp_model_path_pth = f"{model_path_pth}.tmp"
    tmp_model_path_pt = f"{model_path_pt}.tmp"

    # 清理上次异常中断遗留的临时文件，避免后续 replace 失败。
    for tmp_path in (tmp_model_path_pth, tmp_model_path_pt):
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    model.eval()
    export_ok = False

    model_device = next(model.parameters()).device
    try:
        traced_script_module = torch.jit.trace(
            model,
            torch.rand(1, num_channels, args['board_size'], args['board_size'], device=model_device)
        )
        traced_script_module.save(tmp_model_path_pt)
        print(f"TorchScript模型临时文件 {tmp_model_path_pt} 已成功导出。")
        export_ok = True
    except Exception as e:
        print(f"【错误】trace 导出TorchScript模型失败: {e}")

    # trace 失败时尝试 script 兜底，避免双卡流水线因缺少 .pt 长时间空转。
    if not export_ok:
        original_device = model_device
        try:
            model.to(torch.device('cpu'))
            scripted_module = torch.jit.script(model)
            scripted_module.save(tmp_model_path_pt)
            export_ok = True
            print(f"TorchScript模型临时文件 {tmp_model_path_pt} 已通过 script 兜底导出。")
        except Exception as fallback_e:
            print(f"【错误】script 兜底导出也失败: {fallback_e}")
        finally:
            model.to(original_device)

    if not export_ok:
        if os.path.exists(tmp_model_path_pt):
            try:
                os.remove(tmp_model_path_pt)
            except OSError:
                pass
        print("[警告] 本次模型导出失败，已跳过旧模型清理以保留回滚能力。")
        return False

    try:
        torch.save(model.state_dict(), tmp_model_path_pth)
        os.replace(tmp_model_path_pt, model_path_pt)
        os.replace(tmp_model_path_pth, model_path_pth)
        print(f"TorchScript模型 {model_path_pt} 已发布。")
        print(f"模型 {model_path_pth} 已发布。")
    except Exception as publish_e:
        print(f"【错误】模型发布失败: {publish_e}")
        for tmp_path in (tmp_model_path_pth, tmp_model_path_pt):
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return False

    # 仅在本次导出和发布成功时清理旧权重，避免失败时误删可恢复检查点。
    remove_old_models(args, keep_max=3)
    return True


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
                    loaded_self_play = pickle.load(f)

                expected_state_len = self.args['num_channels'] * self.args['board_size'] * self.args['board_size']
                expected_policy_len = self.args['board_size'] * self.args['board_size']
                filtered_self_play = []
                for item in loaded_self_play:
                    if not isinstance(item, (list, tuple)) or len(item) < 3:
                        continue
                    state, policy, value = item[0], item[1], item[2]
                    if len(state) != expected_state_len or len(policy) != expected_policy_len:
                        continue
                    filtered_self_play.append((state, policy, value))

                self.self_play_data.extend(filtered_self_play)
                dropped = len(loaded_self_play) - len(filtered_self_play)
                print(f"[数据加载] 成功从 '{self_play_data_file}' 加载了 {len(filtered_self_play)} 条自对弈数据 (过滤 {dropped} 条无效样本)。")
            except Exception as e:
                print(f"[警告] 加载 '{self_play_data_file}' 失败: {e}")
        human_data_file = 'human_games.pkl'
        if os.path.exists(human_data_file):
            try:
                with open(human_data_file, 'rb') as f:
                    expert_data_loaded = pickle.load(f)

                expected_state_len = self.args['num_channels'] * self.args['board_size'] * self.args['board_size']
                expected_policy_len = self.args['board_size'] * self.args['board_size']
                filtered_expert_data = []
                for item in expert_data_loaded:
                    if not isinstance(item, (list, tuple)) or len(item) < 3:
                        continue
                    state, policy = item[0], item[1]
                    if len(state) != expected_state_len or len(policy) != expected_policy_len:
                        continue
                    # 专家池仅保留带策略监督的样本
                    if not np.any(policy):
                        continue
                    filtered_expert_data.append(item)

                self.expert_data.clear();
                self.expert_data.extend(filtered_expert_data)
                dropped = len(expert_data_loaded) - len(filtered_expert_data)
                print(f"[数据加载] 成功从 '{human_data_file}' 加载了 {len(self.expert_data)} 条专家数据 (过滤 {dropped} 条无效/无标签样本)。")
            except Exception as e:
                print(f"[警告] 加载 '{human_data_file}' 失败: {e}")
        print(f"\n--- 启动时总数据: {len(self.self_play_data)} (自对弈) + {len(self.expert_data)} (专家) ---\n")

        best_model_info = find_latest_model_file()
        if best_model_info is None:
            print("[启动修复] 未检测到模型文件，正在自动创建 model_0...")
            if not save_model(self.model, 0, self.args):
                raise RuntimeError("Coach.learn 启动失败：自动创建 model_0 失败。")
            best_model_info = find_latest_model_file()
            if best_model_info is None:
                raise RuntimeError("Coach.learn 启动失败：创建 model_0 后仍无法检索到模型文件。")

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
            if not os.path.exists(best_model_path_pt):
                print(f"[检测] 本轮基线 .pt 缺失: {best_model_path_pt}，正在补导出...")
                if not save_model(self.model, current_model_epoch, self.args):
                    raise RuntimeError(f"本轮补导出失败: {best_model_path_pt}")
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
            total_batch_size = self.args['batch_size']
            expert_ratio = self.args.get('expert_data_ratio', 0.25)
            expert_batch_size = int(total_batch_size * expert_ratio) if len(self.expert_data) > 0 else 0
            expert_batch_size = min(len(self.expert_data), expert_batch_size)
            required_self_play = total_batch_size - expert_batch_size
            if len(self.self_play_data) < required_self_play:
                print(f"警告：自对弈数据不足({len(self.self_play_data)}/{required_self_play})，跳过本次训练。")
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
                print("\n步骤3.2: 评估候选模型 (车轮战模式)...")
                # 导出候选模型
                candidate_model_path_pt = f"candidate_{cand_idx}.pt"
                cand_model.eval()
                candidate_export_ok = False
                try:
                    traced_script_module = torch.jit.trace(
                        cand_model,
                        torch.rand(1, self.args['num_channels'], self.args['board_size'], self.args['board_size']).to(device)
                    )
                    traced_script_module.save(candidate_model_path_pt)
                    candidate_export_ok = True
                except Exception as trace_e:
                    print(f"[警告] 候选模型 trace 导出失败: {trace_e}，尝试 script 兜底...")
                    original_device = next(cand_model.parameters()).device
                    try:
                        cand_model.to(torch.device('cpu'))
                        scripted_module = torch.jit.script(cand_model)
                        scripted_module.save(candidate_model_path_pt)
                        candidate_export_ok = True
                    except Exception as script_e:
                        print(f"[错误] 候选模型 script 兜底导出失败: {script_e}，跳过该候选。")
                    finally:
                        cand_model.to(original_device)

                if not candidate_export_ok:
                    continue

                # --- 核心改动开始 ---
                # 1. 寻找对手 (最近的3个模型)
                import glob
                # 匹配 model_X.pt (排除 candidate)
                parsed_models = []
                for model_path in glob.glob("model_*.pt"):
                    if "candidate" in model_path:
                        continue
                    epoch = extract_model_epoch(model_path)
                    if epoch is None:
                        print(f"[警告] 跳过无法解析epoch的模型文件: {model_path}")
                        continue
                    parsed_models.append((epoch, model_path))
                parsed_models.sort(key=lambda x: x[0], reverse=True)
                all_models = [model_path for _, model_path in parsed_models]
                opponents = all_models[:3]  # 取最近的3个，例如 [model_10.pt, model_9.pt, model_8.pt]
                if not opponents: opponents = [best_model_path_pt]  # 兜底

                # 2. 准备开局数据
                # 总局数 = 对手数 * 每轮局数。这里我们保持每轮评估局数一致
                games_per_opponent = max(2, int(self.args.get('num_eval_games', 20)))
                if games_per_opponent % 2 != 0:
                    games_per_opponent += 1
                total_initial_states = prepare_eval_states(self.self_play_data, games_per_opponent * len(opponents),
                                                           self.args)

                print(f"  - 准备了 {len(total_initial_states)} 个开局局面 (来自真实数据)")
                print(f"  - 挑战对手: {opponents}")

                pass_all_checks = True
                total_wins, total_games_played = 0, 0
                eval_args = self.args.copy()
                eval_args['num_eval_games'] = games_per_opponent // 2  # 每一方执黑一半
                eval_args['num_eval_simulations'] = self.args['num_searches']

                state_cursor = 0
                for opp_path in opponents:
                    # 切片取出一组开局供本轮使用
                    batch_states = total_initial_states[state_cursor: state_cursor + games_per_opponent]
                    state_cursor += games_per_opponent

                    # 运行评估 (双向)
                    # Mode 2: Candidate(新) 执黑
                    res1 = cpp_mcts_engine.run_parallel_evaluation(opp_path, candidate_model_path_pt,
                                                                   device.type == 'cuda', eval_args, 2,
                                                                   batch_states[:len(batch_states) // 2])
                    # Mode 1: Candidate(新) 执白
                    res2 = cpp_mcts_engine.run_parallel_evaluation(opp_path, candidate_model_path_pt,
                                                                   device.type == 'cuda', eval_args, 1,
                                                                   batch_states[len(batch_states) // 2:])

                    # 统计本轮胜率
                    # model2 是 candidate
                    wins = res1.get("model2_wins", 0) + res2.get("model2_wins", 0)
                    losses = res1.get("model1_wins", 0) + res2.get("model1_wins", 0)
                    draws = res1.get("draws", 0) + res2.get("draws", 0)
                    round_total = wins + losses + draws
                    round_win_rate = wins / round_total if round_total > 0 else 0

                    total_wins += wins
                    total_games_played += round_total

                    # 晋升门槛 (这里硬编码为 0.50，稍后在config中统一)
                    threshold = self.args.get('promotion_win_rate', 0.50)
                    print(f"    vs {opp_path}: 胜率 {round_win_rate:.2%} ({wins}-{losses}-{draws}) | 门槛 {threshold}")

                    if round_win_rate < threshold:
                        pass_all_checks = False

                avg_win_rate = total_wins / total_games_played if total_games_played > 0 else 0
                print(f"  - 综合平均胜率: {avg_win_rate:.2%}")

                if os.path.exists(candidate_model_path_pt): os.remove(candidate_model_path_pt)

                # 判定: 必须所有轮次达标 且 平均胜率显著 > 门槛
                if pass_all_checks and avg_win_rate >= self.args.get('promotion_win_rate', 0.50):
                    # ... (这里保留原有的晋升保存逻辑) ...
                    next_model_epoch = current_model_epoch + 1
                    print(f"【模型晋升】候选 {cand_idx + 1} 全面胜出！保存为 model_{next_model_epoch}。")
                    elo_best_model += 50  # 简单增加Elo
                    previous_best_path = best_model_info['path']
                    self.model.load_state_dict(cand_model.state_dict())
                    if save_model(self.model, next_model_epoch, self.args):
                        current_model_epoch = next_model_epoch
                        promotion_achieved = True
                        break
                    print("【错误】晋升模型导出失败，本轮晋升取消。")
                    try:
                        self.model.load_state_dict(torch.load(previous_best_path, map_location=device))
                        print(f"[回滚] 已恢复为晋升前最优模型: {previous_best_path}")
                    except Exception as rollback_e:
                        raise RuntimeError(f"晋升失败后回滚也失败: {rollback_e}")
                else:
                    print(f"【模型丢弃】未能击败历史模型池。")
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
        total_eval_states = prepare_eval_states(self.self_play_data, total_games, self.args)
        first_half_states = total_eval_states[:games_per_side]
        second_half_states = total_eval_states[games_per_side:games_per_side * 2]
        print(f"\n[实验一] 新模型执黑，进行 {games_per_side} 局...")
        results1 = cpp_mcts_engine.run_parallel_evaluation(model1_pt_path, model2_pt_path, use_gpu, eval_args, 2,
                                   first_half_states)
        new_as_p1_wins, old_as_p2_wins, draws1 = results1.get("model2_wins", 0), results1.get("model1_wins",
                                                                                              0), results1.get("draws",
                                                                                                               0)
        print(f"\n[实验二] 旧模型执黑，进行 {games_per_side} 局...")
        results2 = cpp_mcts_engine.run_parallel_evaluation(model1_pt_path, model2_pt_path, use_gpu, eval_args, 1,
                                   second_half_states)
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
        if not save_model(current_model, 0, args):
            raise RuntimeError("初始模型导出失败，无法继续训练。")
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
                latest_pt_path = latest_model_info['path'].replace('.pth', '.pt')
                if not os.path.exists(latest_pt_path):
                    print(f"[检测] 缺少对应TorchScript模型: {latest_pt_path}，正在重导出...")
                    if not save_model(current_model, latest_model_info['epoch'], args):
                        raise RuntimeError("重导出已有模型的 .pt 失败。")
            except Exception as e:
                print(f"加载权重失败: {e}，将从随机权重开始。")
                recovery_epoch = latest_model_info['epoch'] + 1
                print(f"[入口兜底] 正在导出恢复基线 model_{recovery_epoch} (保留原有 model_{latest_model_info['epoch']})...")
                if not save_model(current_model, recovery_epoch, args):
                    raise RuntimeError("入口兜底导出失败：缺少可用 .pt，后续自对弈将无法启动。")
        else:
            print("模型结构与当前配置不一致，将执行自动迁移学习。")
            try:
                current_model = transfer_weights(current_model, latest_model_info['path'])
                print("为迁移学习后的新模型创建匹配的 .pt 文件...");
                if not save_model(current_model, latest_model_info['epoch'], args):
                    raise RuntimeError("迁移学习模型导出失败，无法继续训练。")
            except Exception as e:
                print(f"迁移学习失败: {e}，将从随机权重开始训练新结构模型。")
                recovery_epoch = latest_model_info['epoch'] + 1
                print(f"[入口兜底] 迁移失败，正在导出新结构恢复基线 model_{recovery_epoch}...")
                if not save_model(current_model, recovery_epoch, args):
                    raise RuntimeError("迁移失败后的恢复模型导出失败，无法继续训练。")

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