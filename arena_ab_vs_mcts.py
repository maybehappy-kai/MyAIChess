
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作用: 评估最新 MCTS 模型与 Alpha-Beta 剪枝 AI 的对战表现（多进程加速版）
使用方法:
    python arena_ab_vs_mcts_mp.py
"""

import os
import re
import sys
import math
import copy
import torch
import multiprocessing as mp
from typing import Tuple, List

import cpp_mcts_engine            # C++ 推理引擎
from config import args           # 你的训练/模型超参

# ==============================================================================
# 0. 工具函数
# ==============================================================================

def find_latest_model_file() -> str:
    """从当前目录中找最新的 model_xxx.pt 文件"""
    pattern = re.compile(r"model_(\d+).*\.pt")
    latest_epoch = -1
    latest_file = None
    for fname in os.listdir("."):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = fname
    return latest_file


# ==============================================================================
# 1. 游戏逻辑（保持与原项目一致）
# ==============================================================================

class GameLogic:
    """纯 Python 的轻量级游戏状态管理器（9x9 夺地简化版）"""
    def __init__(self, board_size: int = 9, num_rounds: int = 25):
        self.board_size = board_size
        self.max_total_moves = num_rounds * 2
        self.reset()

    # ---------- 基础接口 ----------
    def reset(self):
        self.board_pieces   = [[0] * self.board_size for _ in range(self.board_size)]
        self.board_territory= [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        self.current_move_number = 0

    def get_valid_moves(self) -> List[int]:
        moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board_pieces[r][c] == 0 and self.board_territory[r][c] != -self.current_player:
                    moves.append(r * self.board_size + c)
        return moves

    def execute_move(self, action: int):
        r, c = divmod(action, self.board_size)
        player = self.current_player
        self.board_pieces[r][c] = player

        # 三连消除 + 领地更新
        combos = [
            [(0,-2),(0,-1),(0,0)], [(0,-1),(0,0),(0,1)], [(0,0),(0,1),(0,2)],
            [(-2,0),(-1,0),(0,0)], [(-1,0),(0,0),(1,0)], [(0,0),(1,0),(2,0)],
            [(-2,-2),(-1,-1),(0,0)], [(-1,-1),(0,0),(1,1)], [(0,0),(1,1),(2,2)],
            [(2,-2),(1,-1),(0,0)],  [(1,-1),(0,0),(-1,1)],[(0,0),(-1,1),(-2,2)]
        ]
        pieces_to_remove = set()
        axis_found = [False]*4
        for i, combo in enumerate(combos):
            pts = [(r+dr,c+dc) for dr,dc in combo]
            if all(0<=pr<self.board_size and 0<=pc<self.board_size and
                   self.board_pieces[pr][pc]==player for pr,pc in pts):
                pieces_to_remove.update(pts)
                axis_found[i//3] = True

        if pieces_to_remove:
            for pr,pc in pieces_to_remove:
                self.board_pieces[pr][pc] = 0
            directions = [(0,1),(1,0),(1,1),(1,-1)]
            for idx, (dr,dc) in enumerate(directions):
                if not axis_found[idx]:
                    continue
                for sign in (1,-1):
                    cr, cc = r, c
                    while 0<=cr<self.board_size and 0<=cc<self.board_size:
                        if self.board_pieces[cr][cc] == -player:
                            break
                        self.board_territory[cr][cc] = player
                        cr += sign*dr
                        cc += sign*dc

        self.current_move_number += 1
        self.current_player *= -1

    def check_game_end(self) -> Tuple[int,int,bool]:
        if (self.current_move_number >= self.max_total_moves) or (not self.get_valid_moves()):
            p1 = sum(row.count(1)  for row in self.board_territory)
            p2 = sum(row.count(-1) for row in self.board_territory)
            return p1, p2, True
        return 0,0,False


# ==============================================================================
# 2. Alpha-Beta 剪枝 AI（带综合评估）
# ==============================================================================

class AlphaBetaAI:
    def __init__(self, board_size:int, depth:int=2):
        self.board_size = board_size
        self.depth = depth

    # -------- 评估辅助 ----------
    def _count_live_twos(self, board, player)->int:
        cnt = 0
        dirs = [(0,1),(1,0),(1,1),(1,-1)]
        for r in range(self.board_size):
            for c in range(self.board_size):
                for dr,dc in dirs:
                    pts = [(r-dr,c-dc), (r,c), (r+dr,c+dc), (r+2*dr,c+2*dc)]
                    if not all(0<=pr<self.board_size and 0<=pc<self.board_size for pr,pc in pts):
                        continue
                    pattern = [board[pr][pc] for pr,pc in pts]
                    if pattern == [0,player,player,0]:
                        cnt += 1
        return cnt//2  # 双向重复

    def evaluate_board(self, g:GameLogic)->float:
        W_TERRITORY = 1.0
        W_MY_POT    = 10.0
        W_OPP_POT   = 15.0

        p1 = sum(row.count(1)  for row in g.board_territory)
        p2 = sum(row.count(-1) for row in g.board_territory)
        territory_adv = p1 - p2 if g.current_player==1 else p2 - p1

        my_live_twos  = self._count_live_twos(g.board_pieces,  g.current_player)
        opp_live_twos = self._count_live_twos(g.board_pieces, -g.current_player)
        pot_adv = my_live_twos*W_MY_POT - opp_live_twos*W_OPP_POT
        return W_TERRITORY*territory_adv + pot_adv

    # -------- 核心搜索 ----------
    def find_best_move(self, g:GameLogic)->int:
        best_move = -1
        best_val  = -math.inf
        alpha, beta = -math.inf, math.inf
        for mv in g.get_valid_moves():
            tmp = copy.deepcopy(g)
            tmp.execute_move(mv)
            val = -self._alphabeta(tmp, self.depth-1, -beta, -alpha, False)
            if val > best_val:
                best_val, best_move = val, mv
            alpha = max(alpha, best_val)
        return best_move

    def _alphabeta(self, g:GameLogic, depth:int, alpha:float, beta:float, is_max:bool)->float:
        _,_,term = g.check_game_end()
        if depth==0 or term:
            return self.evaluate_board(g)
        if is_max:
            val = -math.inf
            for mv in g.get_valid_moves():
                tmp = copy.deepcopy(g)
                tmp.execute_move(mv)
                val = max(val, self._alphabeta(tmp, depth-1, alpha, beta, False))
                alpha = max(alpha, val)
                if alpha>=beta:
                    break
            return val
        else:
            val = math.inf
            for mv in g.get_valid_moves():
                tmp = copy.deepcopy(g)
                tmp.execute_move(mv)
                val = min(val, self._alphabeta(tmp, depth-1, alpha, beta, True))
                beta = min(beta, val)
                if beta<=alpha:
                    break
            return val


# ==============================================================================
# 3. 单局对战：给多进程调用
# ==============================================================================

def run_single_game(index:int,
                    mcts_player_id:int,
                    mcts_model_file:str,
                    mcts_args:dict,
                    ab_depth:int,
                    board_size:int,
                    num_rounds:int,
                    device_type:str) -> str:
    """
    返回:
        "MCTS", "AB", "Draw"
    """
    game = GameLogic(board_size=board_size, num_rounds=num_rounds)
    ab_ai = AlphaBetaAI(board_size=board_size, depth=ab_depth)

    while True:
        s1,s2,ended = game.check_game_end()
        if ended:
            if s1>s2:
                return "MCTS" if mcts_player_id==1 else "AB"
            elif s2>s1:
                return "MCTS" if mcts_player_id==-1 else "AB"
            else:
                return "Draw"

        if game.current_player == mcts_player_id:
            action = cpp_mcts_engine.find_best_action(
                game.board_pieces,
                game.board_territory,
                game.current_player,
                game.current_move_number,
                mcts_model_file,
                device_type=="cuda",
                mcts_args
            )
        else:
            action = ab_ai.find_best_move(game)

        if action == -1:               # 无棋可走
            return "Draw"
        game.execute_move(action)


# ==============================================================================
# 4. 主入口：多进程调度
# ==============================================================================

def main():
    NUM_GAMES   = 100
    AB_DEPTH    = 3
    MAX_WORKERS = min(mp.cpu_count(), 10)   # 根据机器核数调整
    print(f"并行进程数: {MAX_WORKERS}")

    mcts_model_file = find_latest_model_file()
    if not mcts_model_file:
        print("未找到模型 *.pt，先训练后再评估")
        sys.exit(1)

    # ------- 准备 MCTS 运行参数 -------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcts_args = args.copy()
    hist_ch  = (mcts_args.get('history_steps',0)+1)*4
    meta_ch  = 4
    mcts_args['num_channels']   = hist_ch + meta_ch
    mcts_args['board_size']     = args['board_size']
    mcts_args['max_total_moves']= args['num_rounds']*2

    # ------- 生成任务列表 -------
    tasks = []
    for i in range(NUM_GAMES):
        mcts_player_id = 1 if i%2==0 else -1
        tasks.append( (i, mcts_player_id, mcts_model_file,
                       mcts_args, AB_DEPTH,
                       args['board_size'], args['num_rounds'],
                       device.type) )

    print("="*60)
    print(f"开始并行对战评估，共 {NUM_GAMES} 局…")
    print("="*60)

    # ------- 多进程跑 -------
    mp.set_start_method("spawn", force=True)   # 避免 CUDA/Fork 问题
    with mp.Pool(processes=MAX_WORKERS) as pool:
        results = pool.starmap(run_single_game, tasks)

    # ------- 统计 -------
    mcts_wins = results.count("MCTS")
    ab_wins   = results.count("AB")
    draws     = results.count("Draw")

    print("\n评估完成！")
    print("-"*60)
    print(f"MCTS AI 胜场  : {mcts_wins}")
    print(f"Alpha-Beta 胜场: {ab_wins}")
    print(f"平局          : {draws}")
    total = mcts_wins+ab_wins+draws
    if total:
        print(f"MCTS 胜率     : {mcts_wins/total:.2%}")
    print("="*60)


if __name__ == "__main__":
    main()
