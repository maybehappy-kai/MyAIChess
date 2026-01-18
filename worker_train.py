# file: worker_train.py
# 最终增强版：支持专家数据热重载 (Hot-Reload)
import os
import time
import glob
import pickle
import torch
import torch.optim as optim
import numpy as np
import platform
import ctypes
import re

# --- 关键配置 ---
TRAINER_DEVICE_ID = 1
# ---------------

os.environ["CUDA_VISIBLE_DEVICES"] = str(TRAINER_DEVICE_ID)

import coach
from coach import Coach, save_model, find_latest_model_file, prepare_eval_states
from neural_net import ExtendedConnectNet
from config import args
import cpp_mcts_engine

DATA_BUFFER_DIR = "data_buffer"
SELF_PLAY_DATA_FILE = "self_play_data.pkl"
EXPERT_DATA_FILE = "human_games.pkl"


def clear_windows_memory():
    if platform.system() == "Windows":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass


def load_expert_data(trainer, filepath):
    """辅助函数：安全加载并覆盖专家数据"""
    try:
        with open(filepath, 'rb') as f:
            new_expert_data = pickle.load(f)

        # 覆盖更新：因为 human_games.pkl 通常包含所有历史数据
        trainer.expert_data.clear()
        trainer.expert_data.extend(new_expert_data)
        print(f"\n[专家] 数据已热重载！当前专家池: {len(trainer.expert_data)} 条")
        return True
    except Exception as e:
        print(f"[警告] 热重载专家数据失败: {e}")
        return False


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
    if latest_info:
        print(f"加载已有模型: {latest_info['path']} (Epoch {latest_info['epoch']})")
        model.load_state_dict(torch.load(latest_info['path'], map_location=device))
        current_epoch = latest_info['epoch']
    else:
        print("初始化新模型 model_0...")
        save_model(model, 0, args)
        current_epoch = 0

    trainer = Coach(model, args)
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # 1. 初始加载历史数据
    if os.path.exists(SELF_PLAY_DATA_FILE):
        try:
            with open(SELF_PLAY_DATA_FILE, 'rb') as f:
                trainer.self_play_data.extend(pickle.load(f))
            print(f"[历史] 自对弈数据: {len(trainer.self_play_data)} 条")
        except Exception:
            pass

    # 记录专家文件的最后修改时间
    last_expert_mtime = 0
    if os.path.exists(EXPERT_DATA_FILE):
        last_expert_mtime = os.path.getmtime(EXPERT_DATA_FILE)
        load_expert_data(trainer, EXPERT_DATA_FILE)

    print(f"--- 启动就绪 ---")

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
            except Exception:
                pass

        # --- B. 自对弈数据加载 ---
        pkl_files = sorted(glob.glob(os.path.join(DATA_BUFFER_DIR, "*.pkl")))

        if pkl_files:
            print(f"\n发现 {len(pkl_files)} 个自对弈包...")
            buffer_data = []
            for pkl_f in pkl_files:
                try:
                    with open(pkl_f, 'rb') as f:
                        buffer_data.extend(pickle.load(f))
                    os.remove(pkl_f)
                except Exception:
                    pass

            enable_filtering = args.get('filter_zero_policy_data', True)
            valid_samples = []
            for item in buffer_data:
                # 兼容格式
                state, policy, value = item[0], item[1], item[2]
                if not enable_filtering or np.any(policy):
                    valid_samples.append(item)

            if valid_samples:
                trainer.self_play_data.extend(valid_samples)
                # 立即持久化
                try:
                    with open(SELF_PLAY_DATA_FILE, 'wb') as f:
                        pickle.dump(trainer.self_play_data, f)
                except Exception:
                    pass
                print(f"入库 {len(valid_samples)} 条新数据")

        # 检查数据量
        if len(trainer.self_play_data) < args['batch_size']:
            print(f"\r等待数据... (自对弈:{len(trainer.self_play_data)} | 专家:{len(trainer.expert_data)})", end="")
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
            # 导出候选模型 (保持不变)
            traced = torch.jit.trace(model,
                                     torch.rand(1, args['num_channels'], args['board_size'], args['board_size']).to(
                                         device))
            traced.save(candidate_path)

            best_info = find_latest_model_file()
            # 注意：best_info['path'] 是 .pth，我们需要 .pt
            # 但这里我们要重写逻辑去寻找最近的3个模型

            # 1. 寻找对手 (最近的3个模型)
            # 排除 candidate 和非 .pt 文件 (find_latest_model_file 找的是 pth，我们这里直接扫目录找 pt 更方便用于推理)
            pt_files = glob.glob("model_*.pt")
            # 提取 epoch 数字并排序
            sorted_pts = sorted([f for f in pt_files if "candidate" not in f],
                                key=lambda x: int(re.search(r"model_(\d+)", x).group(1)), reverse=True)

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
                    continue

                games_per_opponent = args.get('num_eval_games', 20) // 2  # 减半，因为要跑3轮
                # 这里的 games_per_opponent 指的是单侧局数，还是总局数？
                # 原逻辑 eval_args['num_eval_games'] = args.get('num_eval_games', 20) // 2
                # 我们保持一致，让 args['num_eval_games'] 代表"每一轮车轮战的总局数"

                total_games_per_round = args.get('num_eval_games', 20)
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
                    res1 = cpp_mcts_engine.run_parallel_evaluation(opp_path, candidate_path, True, eval_args, 2,
                                                                   batch_states[:len(batch_states) // 2])
                    res2 = cpp_mcts_engine.run_parallel_evaluation(opp_path, candidate_path, True, eval_args, 1,
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
                current_epoch += 1
                save_model(model, current_epoch, args)
                clear_windows_memory()
            else:
                print(">>> 晋升失败。")
                # 恢复为当前最好的模型权重，继续训练
                if best_info:
                    model.load_state_dict(torch.load(best_info['path'], map_location=device))

        except Exception as e:
            print(f"[错误] {e}")
            best_info = find_latest_model_file()
            if best_info:
                model.load_state_dict(torch.load(best_info['path'], map_location=device))

        finally:
            if os.path.exists(candidate_path):
                os.remove(candidate_path)


if __name__ == "__main__":
    main()