# file: arena_ab_vs_mcts.py
# 作用: 在一个简单的对战环境中，让最新的MCTS模型与一个基础的Alpha-Beta剪枝AI进行对战，以评估模型性能。
# 版本: 综合评估函数增强版

import sys
import os
import re
import copy
import math
import torch
import cpp_mcts_engine
from config import args
from tqdm import tqdm


# ==============================================================================
# 1. 辅助函数和游戏逻辑类 (保持不变)
# ==============================================================================

def find_latest_model_file():
    """
    从当前目录查找最新的 .pt 模型文件。
    这是 MCTS AI 需要的核心模型。
    """
    path = "."
    max_epoch = -1
    latest_file = None
    pattern = re.compile(r"model_(\d+).*\.pt")
    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = f
    return latest_file


class GameLogic:
    """
    一个纯Python的游戏状态管理器，用于驱动对战。
    其逻辑与 play.py 中的Gomoku类基本一致，用于非GUI环境。
    """

    def __init__(self, board_size=9, num_rounds=25):
        self.board_size = board_size
        self.max_total_moves = num_rounds * 2
        self.reset()

    def reset(self):
        """重置棋盘到初始状态"""
        self.board_pieces = [[0] * self.board_size for _ in range(self.board_size)]
        self.board_territory = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        self.current_move_number = 0

    def get_valid_moves(self):
        """获取所有合法落子点的列表 (返回一个action索引的列表)"""
        valid_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board_pieces[r][c] == 0 and self.board_territory[r][c] != -self.current_player:
                    valid_moves.append(r * self.board_size + c)
        return valid_moves

    def execute_move(self, action):
        """执行一步移动并更新棋盘"""
        r, c = action // self.board_size, action % self.board_size
        player_who_moved = self.current_player
        self.board_pieces[r][c] = player_who_moved

        combinations = [
            [(0, -2), (0, -1), (0, 0)], [(0, -1), (0, 0), (0, 1)], [(0, 0), (0, 1), (0, 2)],
            [(-2, 0), (-1, 0), (0, 0)], [(-1, 0), (0, 0), (1, 0)], [(0, 0), (1, 0), (2, 0)],
            [(-2, -2), (-1, -1), (0, 0)], [(-1, -1), (0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)],
            [(2, -2), (1, -1), (0, 0)], [(1, -1), (0, 0), (-1, 1)], [(0, 0), (-1, 1), (-2, 2)]]

        pieces_to_remove = set()
        axis_found = [False] * 4

        for i, combo in enumerate(combinations):
            points = [(r + dr, c + dc) for dr, dc in combo]
            if all(0 <= pr < self.board_size and 0 <= pc < self.board_size and self.board_pieces[pr][
                pc] == player_who_moved for pr, pc in points):
                pieces_to_remove.update(points)
                axis_found[i // 3] = True

        if pieces_to_remove:
            for pr, pc in pieces_to_remove:
                self.board_pieces[pr][pc] = 0

            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for i in range(4):
                if axis_found[i]:
                    dr, dc = directions[i]
                    for sign in [1, -1]:
                        cr, cc = r, c
                        while 0 <= cr < self.board_size and 0 <= cc < self.board_size:
                            if self.board_pieces[cr][cc] == -player_who_moved: break
                            self.board_territory[cr][cc] = player_who_moved
                            cr += sign * dr
                            cc += sign * dc

        self.current_move_number += 1
        self.current_player *= -1

    def check_game_end(self):
        """检查游戏是否结束，并返回分数和结束标志"""
        if self.current_move_number >= self.max_total_moves or not self.get_valid_moves():
            p1_score = sum(row.count(1) for row in self.board_territory)
            p2_score = sum(row.count(-1) for row in self.board_territory)
            return p1_score, p2_score, True
        return 0, 0, False


# ==============================================================================
# 2. Alpha-Beta 剪枝 AI 实现 (评估函数已修改)
# ==============================================================================

class AlphaBetaAI:
    def __init__(self, board_size, depth=2):
        self.board_size = board_size
        self.depth = depth
        print(f"Alpha-Beta AI 已初始化，搜索深度为: {self.depth}，使用综合评估函数。")

    def _count_live_twos(self, board, player):
        """
        一个辅助函数，用于计算指定玩家在棋盘上的“活二”数量。
        “活二”定义为：[空, 子, 子, 空]
        """
        count = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平, 竖直, 主对角线, 副对角线

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dr, dc in directions:
                    # 检查模式 [空, 子, 子, 空]
                    p1_r, p1_c = r - dr, c - dc
                    p2_r, p2_c = r, c
                    p3_r, p3_c = r + dr, c + dc
                    p4_r, p4_c = r + 2 * dr, c + 2 * dc

                    # 确保所有点都在棋盘内
                    points = [(p1_r, p1_c), (p2_r, p2_c), (p3_r, p3_c), (p4_r, p4_c)]
                    if not all(0 <= pr < self.board_size and 0 <= pc < self.board_size for pr, pc in points):
                        continue

                    # 检查棋子模式
                    pattern = [board[p[0]][p[1]] for p in points]
                    if pattern == [0, player, player, 0]:
                        count += 1
        # 因为每个模式会被从两个方向检测到，所以结果除以2
        return count // 2

    def evaluate_board(self, game_state):
        """
        修改后的综合评估函数。
        评估值 = w1 * 领地优势 + w2 * 我方潜力 - w3 * 敌方潜力
        """
        # 权重系数，可以调整
        W_TERRITORY = 1.0
        W_MY_POTENTIAL = 10.0
        W_OPPONENT_POTENTIAL = 15.0

        # --- 1. 领地优势评估 ---
        p1_score = sum(row.count(1) for row in game_state.board_territory)
        p2_score = sum(row.count(-1) for row in game_state.board_territory)

        if game_state.current_player == 1:
            territory_advantage = p1_score - p2_score
        else:
            territory_advantage = p2_score - p1_score

        # --- 2. 棋形潜力评估 (活二数量) ---
        my_player_id = game_state.current_player
        opponent_player_id = -my_player_id

        my_live_twos = self._count_live_twos(game_state.board_pieces, my_player_id)
        opponent_live_twos = self._count_live_twos(game_state.board_pieces, opponent_player_id)

        potential_advantage = my_live_twos * W_MY_POTENTIAL - opponent_live_twos * W_OPPONENT_POTENTIAL

        # --- 3. 返回综合得分 ---
        final_score = W_TERRITORY * territory_advantage + potential_advantage
        return final_score

    def find_best_move(self, game_state):
        """
        为当前局面寻找最佳移动 (此函数逻辑不变)。
        """
        best_move = -1
        best_value = -math.inf

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return -1

        alpha = -math.inf
        beta = math.inf

        for move in valid_moves:
            temp_game = copy.deepcopy(game_state)
            temp_game.execute_move(move)
            board_value = -self._alphabeta(temp_game, self.depth - 1, -beta, -alpha, False)

            if board_value > best_value:
                best_value = board_value
                best_move = move

            alpha = max(alpha, best_value)

        return best_move

    def _alphabeta(self, game_state, depth, alpha, beta, is_maximizing_player):
        """
        Alpha-Beta 剪枝核心递归函数 (此函数逻辑不变)。
        """
        _, _, is_terminal = game_state.check_game_end()
        if depth == 0 or is_terminal:
            return self.evaluate_board(game_state)

        valid_moves = game_state.get_valid_moves()

        if is_maximizing_player:
            value = -math.inf
            for move in valid_moves:
                temp_game = copy.deepcopy(game_state)
                temp_game.execute_move(move)
                value = max(value, self._alphabeta(temp_game, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for move in valid_moves:
                temp_game = copy.deepcopy(game_state)
                temp_game.execute_move(move)
                value = min(value, self._alphabeta(temp_game, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value


# ==============================================================================
# 3. 对战主程序 (保持不变)
# ==============================================================================

if __name__ == "__main__":
    # --- 初始化 ---
    NUM_GAMES = 20
    mcts_model_file = find_latest_model_file()

    if mcts_model_file is None:
        print("错误：在当前目录下未找到任何 '.pt' 模型文件。请先运行 coach.py 进行训练。")
        sys.exit(1)

    print("=" * 60)
    print("MyAIChess - Alpha-Beta vs MCTS 对战评估")
    print(f"对战总局数: {NUM_GAMES}")
    print(f"MCTS AI 使用的模型: {mcts_model_file}")
    print("=" * 60)

    # 实例化 Alpha-Beta AI
    ab_ai = AlphaBetaAI(board_size=args['board_size'], depth=3)

    # MCTS AI 参数准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcts_args = args.copy()
    history_channels = (mcts_args.get('history_steps', 0) + 1) * 4
    meta_channels = 4
    mcts_args['num_channels'] = history_channels + meta_channels
    mcts_args['board_size'] = args['board_size']
    mcts_args['max_total_moves'] = args['num_rounds'] * 2

    # 战绩记录
    mcts_wins = 0
    ab_wins = 0
    draws = 0

    # --- 开始对战循环 ---
    for i in tqdm(range(NUM_GAMES), desc="对战进度"):
        game = GameLogic(board_size=args['board_size'], num_rounds=args['num_rounds'])

        mcts_player_id = 1 if i % 2 == 0 else -1

        if mcts_player_id == 1:
            tqdm.write(f"\n--- 第 {i + 1}/{NUM_GAMES} 局开始: MCTS (黑) vs Alpha-Beta (白) ---")
        else:
            tqdm.write(f"\n--- 第 {i + 1}/{NUM_GAMES} 局开始: Alpha-Beta (黑) vs MCTS (白) ---")

        while True:
            s1, s2, is_ended = game.check_game_end()
            if is_ended:
                winner_text = ""
                if s1 > s2:
                    if mcts_player_id == 1:
                        mcts_wins += 1; winner_text = "MCTS AI 获胜"
                    else:
                        ab_wins += 1; winner_text = "Alpha-Beta AI 获胜"
                elif s2 > s1:
                    if mcts_player_id == -1:
                        mcts_wins += 1; winner_text = "MCTS AI 获胜"
                    else:
                        ab_wins += 1; winner_text = "Alpha-Beta AI 获胜"
                else:
                    draws += 1
                    winner_text = "平局"
                tqdm.write(f"对局结束。结果: {winner_text} (分数 {s1}:{s2})")
                break

            action = -1
            if game.current_player == mcts_player_id:
                action = cpp_mcts_engine.find_best_action(
                    game.board_pieces, game.board_territory, game.current_player,
                    game.current_move_number, mcts_model_file, device.type == 'cuda', mcts_args
                )
            else:
                action = ab_ai.find_best_move(game)

            if action == -1:
                tqdm.write("警告: AI未能选择有效动作，可能游戏已进入无棋可走的状态。")
                continue

            game.execute_move(action)

    # --- 报告最终结果 ---
    print("\n" + "=" * 60)
    print("所有对局完成！最终战报:")
    print("-" * 60)
    print(f"MCTS AI 胜场: {mcts_wins}")
    print(f"Alpha-Beta AI 胜场: {ab_wins}")
    print(f"平局: {draws}")
    print("-" * 60)

    total_games_played = mcts_wins + ab_wins + draws
    if total_games_played > 0:
        mcts_win_rate = mcts_wins / total_games_played
        print(f"MCTS AI 综合胜率: {mcts_win_rate:.2%}")
    else:
        print("没有进行任何有效对局。")
    print("=" * 60)