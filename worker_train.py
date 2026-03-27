# file: worker_train.py
# 最终增强版：支持专家数据热重载 (Hot-Reload)
import os
import time
import glob
import pickle
import numpy as np
import platform
import ctypes
import re

# --- 关键配置 ---
TRAINER_DEVICE_ID = 1
# ---------------

os.environ["CUDA_VISIBLE_DEVICES"] = str(TRAINER_DEVICE_ID)

import torch
import torch.optim as optim

import coach
from coach import Coach, save_model, find_latest_model_file, prepare_eval_states
from neural_net import ExtendedConnectNet
from config import args
import cpp_mcts_engine

DATA_BUFFER_DIR = "data_buffer"
SELF_PLAY_DATA_FILE = "self_play_data.pkl"
EXPERT_DATA_FILE = "human_games.pkl"
STABLE_PACKET_SUFFIX = ".ready.pkl"


def clear_windows_memory():
    if platform.system() == "Windows":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception as e:
            print(f"[警告] 清理Windows工作集失败: {e}")


def load_expert_data(trainer, filepath):
    """辅助函数：安全加载并覆盖专家数据"""
    try:
        with open(filepath, 'rb') as f:
            new_expert_data = pickle.load(f)

        expected_state_len = args['num_channels'] * args['board_size'] * args['board_size']
        expected_policy_len = args['board_size'] * args['board_size']
        filtered_data = []
        for item in new_expert_data:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            state, policy = item[0], item[1]
            if len(state) != expected_state_len or len(policy) != expected_policy_len:
                continue
            # 专家池仅保留带策略监督的样本
            if not np.any(policy):
                continue
            filtered_data.append(item)

        # 覆盖更新：因为 human_games.pkl 通常包含所有历史数据
        trainer.expert_data.clear()
        trainer.expert_data.extend(filtered_data)
        dropped = len(new_expert_data) - len(filtered_data)
        print(f"\n[专家] 数据已热重载！当前专家池: {len(trainer.expert_data)} 条 (过滤 {dropped} 条无效/无标签样本)")
        return True
    except Exception as e:
        print(f"[警告] 热重载专家数据失败: {e}")
        return False


def load_packet_with_retry(filepath, retries=1, retry_delay=0.2):
    """读取数据包并在短暂延迟后重试一次，降低并发I/O抖动造成的误判。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            with open(filepath, 'rb') as f:
                return True, pickle.load(f), None
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_delay)
    return False, None, last_err


def extract_model_epoch(filename):
    """从模型文件名中提取 epoch，失败时返回 None。"""
    match = re.search(r"model_(\d+)", os.path.basename(filename))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def main():
    print(f"--- 启动训练 Worker (GPU {TRAINER_DEVICE_ID}) ---")
    print(f"监听数据目录: {DATA_BUFFER_DIR}/")
    print(f"监听专家文件: {EXPERT_DATA_FILE}")

    # >>>>>>>>【新增】核心修复代码开始 >>>>>>>>
    # 动态计算输入通道数：(历史步数 + 当前步) * 4个特征平面 + 4个元数据平面
    args['num_channels'] = (args.get('history_steps', 0) + 1) * 4 + 4
    # <<<<<<<<【新增】核心修复代码结束 <<<<<<<<

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ExtendedConnectNet(
        board_size=args['board_size'],
        num_res_blocks=args['num_res_blocks'],
        num_hidden=args['num_hidden'],
        num_channels=args['num_channels']
    ).to(device)

    latest_info = find_latest_model_file()
    startup_loaded_ok = False
    if latest_info:
        print(f"加载已有模型: {latest_info['path']} (Epoch {latest_info['epoch']})")
        current_epoch = latest_info['epoch']
        config_blocks = args['num_res_blocks']
        config_hidden = args['num_hidden']
        config_channels = args['num_channels']
        is_same_architecture = (
            latest_info.get('res_blocks') == config_blocks and
            latest_info.get('hidden_units') == config_hidden and
            latest_info.get('channels') == config_channels
        )

        if is_same_architecture:
            print("模型结构与当前配置一致，直接加载权重继续训练。")
            try:
                model.load_state_dict(torch.load(latest_info['path'], map_location=device))
                startup_loaded_ok = True
                print("权重加载成功！")
            except Exception as e:
                print(f"[警告] 加载权重失败: {e}，将以随机初始化权重继续。")
        else:
            print("模型结构与当前配置不一致，将执行自动迁移学习。")
            try:
                model = coach.transfer_weights(model, latest_info['path'])
                startup_loaded_ok = True
                print("迁移学习完成，将继续训练。")
            except Exception as e:
                print(f"[警告] 迁移学习失败: {e}，将以随机初始化权重继续。")

        latest_pt = latest_info['path'].replace('.pth', '.pt')
        if startup_loaded_ok and not os.path.exists(latest_pt):
            print(f"[检测] 缺少对应TorchScript模型: {latest_pt}，正在重导出...")
            if not save_model(model, current_epoch, args):
                raise RuntimeError("重导出已有模型的 .pt 失败，无法启动双卡流水线。")
        elif not startup_loaded_ok:
            print(f"[提示] 将从随机初始化开始训练 (基准epoch={current_epoch})。")
            recovery_epoch = current_epoch + 1
            print(f"[启动兜底] 正在导出恢复基线模型 model_{recovery_epoch} (保留原有 model_{current_epoch})，避免自对弈端无模型等待...")
            if not save_model(model, recovery_epoch, args):
                raise RuntimeError("启动兜底导出失败：无法生成可用 .pt，双卡流水线可能互相等待。")
            current_epoch = recovery_epoch
    else:
        print("初始化新模型 model_0...")
        if not save_model(model, 0, args):
            raise RuntimeError("初始模型导出失败，无法启动双卡流水线。")
        current_epoch = 0

    trainer = Coach(model, args)
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # 1. 初始加载历史数据
    if os.path.exists(SELF_PLAY_DATA_FILE):
        try:
            with open(SELF_PLAY_DATA_FILE, 'rb') as f:
                loaded_self_play = pickle.load(f)

            expected_state_len = args['num_channels'] * args['board_size'] * args['board_size']
            expected_policy_len = args['board_size'] * args['board_size']
            filtered_self_play = []
            for item in loaded_self_play:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                state, policy, value = item[0], item[1], item[2]
                if len(state) != expected_state_len or len(policy) != expected_policy_len:
                    continue
                filtered_self_play.append((state, policy, value))

            trainer.self_play_data.extend(filtered_self_play)
            dropped = len(loaded_self_play) - len(filtered_self_play)
            print(f"[历史] 自对弈数据: {len(filtered_self_play)} 条 (过滤 {dropped} 条无效样本)")
        except Exception as e:
            print(f"[警告] 读取历史自对弈缓存失败: {e}")

    # 记录专家文件的最后修改时间
    last_expert_mtime = 0
    if os.path.exists(EXPERT_DATA_FILE):
        last_expert_mtime = os.path.getmtime(EXPERT_DATA_FILE)
        load_expert_data(trainer, EXPERT_DATA_FILE)

    print(f"--- 启动就绪 ---")
    packet_fail_counts = {}

    while True:
        # --- A. 专家数据热重载检查 ---
        if os.path.exists(EXPERT_DATA_FILE):
            try:
                current_mtime = os.path.getmtime(EXPERT_DATA_FILE)
                # 如果文件被修改过（时间戳变大），则重载
                if current_mtime > last_expert_mtime:
                    print(f"\n[检测] 发现专家数据文件更新...")
                    # 稍微等待一下，确保 play_pixel_art.py 写完文件
                    time.sleep(1)
                    if load_expert_data(trainer, EXPERT_DATA_FILE):
                        last_expert_mtime = current_mtime
            except Exception as e:
                print(f"[警告] 专家文件监控异常: {e}")

        # --- B. 自对弈数据加载 ---
        stable_files = sorted(glob.glob(os.path.join(DATA_BUFFER_DIR, f"*{STABLE_PACKET_SUFFIX}")))
        # 兼容旧版本落盘格式；新版本应始终使用 *.ready.pkl。
        legacy_files = sorted(
            f for f in glob.glob(os.path.join(DATA_BUFFER_DIR, "*.pkl"))
            if not f.endswith(STABLE_PACKET_SUFFIX)
        )
        pkl_files = stable_files + legacy_files

        if pkl_files:
            print(f"\n发现 {len(pkl_files)} 个自对弈包...")
            buffer_data = []
            for pkl_f in pkl_files:
                ok, packet, err = load_packet_with_retry(pkl_f, retries=1, retry_delay=0.2)
                if ok:
                    buffer_data.extend(packet)
                    packet_fail_counts.pop(pkl_f, None)
                    try:
                        os.remove(pkl_f)
                    except Exception as remove_err:
                        print(f"[警告] 删除已消费数据包失败: {pkl_f} | {remove_err}")
                    continue

                fail_count = packet_fail_counts.get(pkl_f, 0) + 1
                packet_fail_counts[pkl_f] = fail_count
                print(f"[警告] 读取数据包失败(第{fail_count}次)，本轮跳过: {pkl_f} | {err}")

                if fail_count >= 3 and os.path.exists(pkl_f):
                    bad_path = f"{pkl_f}.bad"
                    try:
                        os.replace(pkl_f, bad_path)
                        packet_fail_counts.pop(pkl_f, None)
                        print(f"[警告] 数据包连续失败，已隔离为: {bad_path}")
                    except Exception as quarantine_err:
                        print(f"[警告] 隔离失败数据包失败: {pkl_f} | {quarantine_err}")

            enable_filtering = args.get('filter_zero_policy_data', True)
            valid_samples = []
            expected_state_len = args['num_channels'] * args['board_size'] * args['board_size']
            expected_policy_len = args['board_size'] * args['board_size']
            for item in buffer_data:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                state, policy, value = item[0], item[1], item[2]
                if len(state) != expected_state_len or len(policy) != expected_policy_len:
                    continue
                if not enable_filtering or np.any(policy):
                    valid_samples.append((state, policy, value))

            if valid_samples:
                trainer.self_play_data.extend(valid_samples)
                # 立即持久化
                try:
                    with open(SELF_PLAY_DATA_FILE, 'wb') as f:
                        pickle.dump(trainer.self_play_data, f)
                except Exception as e:
                    print(f"[警告] 持久化自对弈缓存失败: {e}")
                print(f"入库 {len(valid_samples)} 条新数据")

        # 检查数据量（确保自对弈样本足以构成训练批次）
        total_batch_size = args['batch_size']
        expert_ratio = args.get('expert_data_ratio', 0.25)
        expert_batch_size = int(total_batch_size * expert_ratio) if len(trainer.expert_data) > 0 else 0
        expert_batch_size = min(len(trainer.expert_data), expert_batch_size)
        required_self_play = total_batch_size - expert_batch_size
        if len(trainer.self_play_data) < required_self_play:
            print(f"\r等待数据... (自对弈:{len(trainer.self_play_data)}/{required_self_play} | 专家:{len(trainer.expert_data)})", end="")
            time.sleep(5)
            continue

        # --- C. 训练 ---
        print(f"\n>>> [Epoch {current_epoch + 1}] 开始训练...")
        policy_loss, value_loss = trainer.train(model, optimizer)
        print(f"Loss: P={policy_loss:.4f}, V={value_loss:.4f}")

        # --- D. 评估与晋升 (车轮战模式) ---
        candidate_path = "candidate_model.pt"
        try:
            model.eval()
            candidate_export_ok = False
            try:
                traced = torch.jit.trace(
                    model,
                    torch.rand(1, args['num_channels'], args['board_size'], args['board_size']).to(device)
                )
                traced.save(candidate_path)
                candidate_export_ok = True
            except Exception as trace_e:
                print(f"[警告] 候选模型 trace 导出失败: {trace_e}，尝试 script 兜底...")
                original_device = next(model.parameters()).device
                try:
                    model.to(torch.device('cpu'))
                    scripted = torch.jit.script(model)
                    scripted.save(candidate_path)
                    candidate_export_ok = True
                except Exception as script_e:
                    print(f"[错误] 候选模型 script 兜底导出失败: {script_e}")
                finally:
                    model.to(original_device)

            if not candidate_export_ok:
                raise RuntimeError("候选模型导出失败，已跳过本轮评估。")

            best_info = find_latest_model_file()
            # 注意：best_info['path'] 是 .pth，我们需要 .pt
            # 但这里我们要重写逻辑去寻找最近的3个模型

            # 1. 寻找对手 (最近的3个模型)
            # 排除 candidate 和非 .pt 文件 (find_latest_model_file 找的是 pth，我们这里直接扫目录找 pt 更方便用于推理)
            pt_files = glob.glob("model_*.pt")
            parsed_pts = []
            for model_path in pt_files:
                if "candidate" in model_path:
                    continue
                epoch = extract_model_epoch(model_path)
                if epoch is None:
                    print(f"[警告] 跳过无法解析epoch的模型文件: {model_path}")
                    continue
                parsed_pts.append((epoch, model_path))
            parsed_pts.sort(key=lambda x: x[0], reverse=True)
            sorted_pts = [model_path for _, model_path in parsed_pts]

            opponents = sorted_pts[:3]
            # 如果没找到任何 .pt (可能是第一次运行)，就用刚导出的 candidate 自己打自己，或者跳过
            if not opponents:
                print("未找到历史模型，跳过评估，直接晋升 (首轮)。")
                win_rate = 1.0
                pass_all_checks = True
                avg_win_rate = 1.0
            else:
                print(f"评估对手: {opponents}")

                # 2. 准备开局数据
                # 确保有足够的数据用于抽取
                if len(trainer.self_play_data) < 10:
                    print("数据不足，无法抽取开局，跳过评估。")
                    if best_info:
                        model.load_state_dict(torch.load(best_info['path'], map_location=device))
                        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'],
                                               weight_decay=args['weight_decay'])
                        print("[回滚] 评估前置条件不足，已恢复到当前最优模型并重建优化器。")
                    continue

                total_games_per_round = max(2, int(args.get('num_eval_games', 20)))
                if total_games_per_round % 2 != 0:
                    total_games_per_round += 1
                eval_args = args.copy()
                eval_args['num_eval_games'] = total_games_per_round // 2  # 单侧局数
                eval_args['num_eval_simulations'] = args['num_searches']

                # 准备所有轮次需要的开局状态
                total_initial_states = prepare_eval_states(trainer.self_play_data,
                                                           total_games_per_round * len(opponents), args)

                pass_all_checks = True
                total_wins, total_games = 0, 0
                state_cursor = 0

                for opp_path in opponents:
                    # 取出本轮开局
                    batch_states = total_initial_states[state_cursor: state_cursor + total_games_per_round]
                    state_cursor += total_games_per_round

                    # 运行评估 (调用C++新接口)
                    res1 = cpp_mcts_engine.run_parallel_evaluation(opp_path, candidate_path, device.type == 'cuda', eval_args, 2,
                                                                   batch_states[:len(batch_states) // 2])
                    res2 = cpp_mcts_engine.run_parallel_evaluation(opp_path, candidate_path, device.type == 'cuda', eval_args, 1,
                                                                   batch_states[len(batch_states) // 2:])

                    wins = res1.get("model2_wins", 0) + res2.get("model2_wins", 0)
                    losses = res1.get("model1_wins", 0) + res2.get("model1_wins", 0)
                    draws = res1.get("draws", 0) + res2.get("draws", 0)
                    round_total = wins + losses + draws

                    if round_total == 0:
                        round_win_rate = 0
                    else:
                        round_win_rate = wins / round_total

                    total_wins += wins
                    total_games += round_total

                    threshold = args.get('promotion_win_rate', 0.50)
                    print(f"  vs {opp_path}: 胜率 {round_win_rate:.2%} ({wins}-{losses}-{draws})")

                    if round_win_rate < threshold:
                        pass_all_checks = False

                avg_win_rate = total_wins / total_games if total_games > 0 else 0
                print(f"综合胜率: {avg_win_rate:.2%} (Pass All: {pass_all_checks})")

            # 3. 判定晋升
            threshold = args.get('promotion_win_rate', 0.50)
            if pass_all_checks and avg_win_rate >= threshold:
                print(">>> 晋升成功！")
                next_epoch = current_epoch + 1
                if save_model(model, next_epoch, args):
                    current_epoch = next_epoch
                    clear_windows_memory()
                else:
                    print("[错误] 晋升模型导出失败，本次晋升已取消。")
                    if best_info:
                        model.load_state_dict(torch.load(best_info['path'], map_location=device))
                        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'],
                                               weight_decay=args['weight_decay'])
            else:
                print(">>> 晋升失败。")
                # 恢复为当前最好的模型权重，继续训练
                if best_info:
                    model.load_state_dict(torch.load(best_info['path'], map_location=device))
                    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'],
                                           weight_decay=args['weight_decay'])
                    print("[回滚] 已恢复模型并重建优化器状态。")

        except Exception as e:
            print(f"[错误] {e}")
            best_info = find_latest_model_file()
            if best_info:
                model.load_state_dict(torch.load(best_info['path'], map_location=device))
                optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'],
                                       weight_decay=args['weight_decay'])
                print("[回滚] 异常后已恢复模型并重建优化器状态。")

        finally:
            if os.path.exists(candidate_path):
                os.remove(candidate_path)


if __name__ == "__main__":
    main()