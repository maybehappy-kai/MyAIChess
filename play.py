# file: play.py (一个简化且能稳定运行的版本)
import pygame
import torch
import sys
import os
import re
import cpp_mcts_engine
from config import args


def find_latest_model_file():
    """查找最新的 .pt 模型文件"""
    path = "."
    max_epoch = -1
    latest_file = None
    # 正则表达式只寻找 C++ 引擎能用的 .pt 文件
    pattern = re.compile(r"model_(\d+).*\.pt")
    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = f
    return latest_file


class Gomoku:
    """一个纯Python的、用于GUI的游戏状态管理器"""

    def __init__(self, board_size=9, num_rounds=25):
        self.board_size = board_size
        self.max_total_moves = num_rounds * 2
        self.reset()

    def reset(self):
        self.board_pieces = [[0] * self.board_size for _ in range(self.board_size)]
        self.board_territory = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        self.current_move_number = 0

    def get_valid_moves(self):
        """获取所有合法落子点的列表"""
        valid_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                # 只要棋盘上是空的，并且不是对方的领地，就可以下
                if self.board_pieces[r][c] == 0 and self.board_territory[r][c] != -self.current_player:
                    valid_moves.append(r * self.board_size + c)
        return valid_moves

    def execute_move(self, action):
        """执行一步移动并更新棋盘"""
        if not (action in self.get_valid_moves()):
            return False  # 非法移动

        r, c = action // self.board_size, action % self.board_size
        player_who_moved = self.current_player
        self.board_pieces[r][c] = player_who_moved

        # --- 提子和领地计算的简化逻辑 ---
        # 检查是否形成三子
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
        return True

    def check_game_end(self):
        """检查游戏是否结束，并返回分数和结束标志"""
        if self.current_move_number >= self.max_total_moves:
            p1_score = sum(row.count(1) for row in self.board_territory)
            p2_score = sum(row.count(-1) for row in self.board_territory)
            return p1_score, p2_score, True
        return 0, 0, False


class SimpleGameGUI:
    def __init__(self):
        pygame.init()
        self.board_size = args['board_size']
        self.cell_size = 60
        screen_dim = self.board_size * self.cell_size + 200  # 为右侧信息栏留出空间
        self.screen = pygame.display.set_mode((screen_dim, self.board_size * self.cell_size))
        pygame.display.set_caption("MyAIChess - 简约对弈版")

        self.font = pygame.font.SysFont("simhei", 32)
        self.clock = pygame.time.Clock()
        self.game = Gomoku(board_size=self.board_size, num_rounds=args['num_rounds'])

        # --- AI 初始化 ---
        self.model_file = find_latest_model_file()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"欢迎来到 MyAIChess！\n人类玩家执黑(P1)，AI执白(P2)。")
        if self.model_file:
            print(f"AI正在使用模型: {self.model_file}")
        else:
            print("警告：未找到任何模型文件，AI将无法行动！")

    def draw_board(self):
        """绘制棋盘、棋子和右侧信息"""
        self.screen.fill((200, 200, 200))  # 背景色
        board_width = self.board_size * self.cell_size

        # 绘制棋盘网格
        for i in range(self.board_size + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (i * self.cell_size, 0), (i * self.cell_size, board_width))
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * self.cell_size), (board_width, i * self.cell_size))

        # 绘制领地
        for r in range(self.board_size):
            for c in range(self.board_size):
                territory = self.game.board_territory[r][c]
                if territory != 0:
                    color = (255, 200, 200) if territory == 1 else (200, 200, 255)
                    pygame.draw.rect(self.screen, color,
                                     (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

        # 绘制棋子
        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.game.board_pieces[r][c]
                if piece != 0:
                    center = (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)
                    radius = self.cell_size // 2 - 5
                    color = (0, 0, 0) if piece == 1 else (255, 255, 255)
                    pygame.draw.circle(self.screen, color, center, radius)

        # 绘制右侧信息栏
        info_x = board_width + 10
        pygame.draw.rect(self.screen, (240, 240, 240), (board_width, 0, 200, board_width))

        p1_score = sum(row.count(1) for row in self.game.board_territory)
        p2_score = sum(row.count(-1) for row in self.game.board_territory)

        turn_text = "你的回合 (黑)" if self.game.current_player == 1 else "AI回合 (白)"
        text_surface = self.font.render(turn_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (info_x, 20))

        score_text = f"分数: 黑 {p1_score} - 白 {p2_score}"
        text_surface = self.font.render(score_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (info_x, 70))

        moves_left = self.game.max_total_moves - self.game.current_move_number
        moves_text = f"剩余步数: {moves_left}"
        text_surface = self.font.render(moves_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (info_x, 120))

    def run(self):
        """主游戏循环"""
        game_over = False
        winner_text = ""

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # 人类玩家的回合
                if not game_over and self.game.current_player == 1 and event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    c = x // self.cell_size
                    r = y // self.cell_size

                    if 0 <= r < self.board_size and 0 <= c < self.board_size:
                        action = r * self.board_size + c
                        if self.game.execute_move(action):
                            # 玩家下完棋，检查游戏是否结束
                            s1, s2, is_ended = self.game.check_game_end()
                            if is_ended:
                                game_over = True
                                winner_text = "平局" if s1 == s2 else ("你赢了！" if s1 > s2 else "AI赢了！")

            # AI 的回合 (同步阻塞执行)
            if not game_over and self.game.current_player == -1:
                self.draw_board()  # 先刷新一下棋盘，显示“AI回合”
                pygame.display.flip()

                if not self.model_file:
                    print("AI没有模型，无法行动。")
                    game_over = True
                    winner_text = "错误：AI无模型"
                else:
                    print("AI 正在思考...")

                    # --- 核心修复：准备给C++的参数字典 ---
                    ai_args = args.copy()
                    # 1. 从 coach.py 复制过来的参数计算逻辑
                    history_channels = (ai_args.get('history_steps', 0) + 1) * 4
                    meta_channels = 4
                    ai_args['num_channels'] = history_channels + meta_channels
                    # 2. 补充C++引擎需要的其他游戏参数
                    ai_args['board_size'] = self.game.board_size
                    ai_args['max_total_moves'] = self.game.max_total_moves

                    # 调用 C++ 引擎
                    ai_action = cpp_mcts_engine.find_best_action(
                        self.game.board_pieces,
                        self.game.board_territory,
                        self.game.current_player,
                        self.game.current_move_number,
                        self.model_file,
                        self.device.type == 'cuda',
                        ai_args
                    )

                    print(f"AI 选择了动作: {ai_action}")
                    self.game.execute_move(ai_action)

                    # AI下完棋，检查游戏是否结束
                    s1, s2, is_ended = self.game.check_game_end()
                    if is_ended:
                        game_over = True
                        winner_text = "平局" if s1 == s2 else ("你赢了！" if s1 > s2 else "AI赢了！")

            # 绘制与刷新
            self.draw_board()

            if game_over:
                text_surface = self.font.render(winner_text, True, (255, 0, 0))
                text_rect = text_surface.get_rect(
                    center=(self.board_size * self.cell_size / 2, self.board_size * self.cell_size / 2))
                pygame.draw.rect(self.screen, (255, 255, 0), text_rect.inflate(20, 20))
                self.screen.blit(text_surface, text_rect)

            pygame.display.flip()
            self.clock.tick(30)


if __name__ == '__main__':
    game_gui = SimpleGameGUI()
    game_gui.run()