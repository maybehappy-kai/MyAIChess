import pygame
import torch
import sys
import os
import re
import cpp_mcts_engine
from config import args
import random
import math
import threading
import copy
import queue  # ★★★ 新增：导入线程安全的队列模块 ★★★


def find_latest_model_file():
    path = ".";
    max_epoch = -1;
    latest_file = None
    # --- 修改正则表达式以匹配 "model_数字_任意字符.pt" ---
    pattern = re.compile(r"model_(\d+).*\.pt")
    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch: max_epoch = epoch; latest_file = f
    return latest_file


class PythonGomoku:
    def __init__(self, board_size=9, num_rounds=25):
        self.board_size = board_size;
        self.max_total_moves = num_rounds * 2;
        self.last_move = None;
        self.reset()

    def reset(self):
        self.board_pieces = [[0] * self.board_size for _ in range(self.board_size)];
        self.board_territory = [[0] * self.board_size for _ in range(self.board_size)];
        self.current_player = 1;
        self.current_move_number = 0;
        self.last_move = None

    def get_valid_moves(self):
        valid_moves = [False] * (self.board_size * self.board_size)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board_pieces[r][c] == 0 and self.board_territory[r][c] != -self.current_player: valid_moves[
                    r * self.board_size + c] = True
        return valid_moves

    def execute_move(self, action):
        if not self.get_valid_moves(): return False, [], 0
        r, c = action // self.board_size, action % self.board_size
        player_who_moved = self.current_player
        self.board_pieces[r][c] = player_who_moved;
        self.last_move = (r, c);
        pieces_to_remove = set();
        axis_found = [False] * 4
        combinations = [[(0, -2), (0, -1), (0, 0)], [(0, -1), (0, 0), (0, 1)], [(0, 0), (0, 1), (0, 2)],
                        [(-2, 0), (-1, 0), (0, 0)], [(-1, 0), (0, 0), (1, 0)], [(0, 0), (1, 0), (2, 0)],
                        [(-2, -2), (-1, -1), (0, 0)], [(-1, -1), (0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)],
                        [(2, -2), (1, -1), (0, 0)], [(1, -1), (0, 0), (-1, 1)], [(0, 0), (-1, 1), (-2, 2)]]
        line_centers = []
        for i, combo in enumerate(combinations):
            points = [(r + dr, c + dc) for dr, dc in combo]
            if all(0 <= pr < self.board_size and 0 <= pc < self.board_size and self.board_pieces[pr][
                pc] == player_who_moved for pr, pc in points):
                pieces_to_remove.update(points);
                axis_found[i // 3] = True;
                line_centers.append(points[1])
        if pieces_to_remove:
            for pr, pc in pieces_to_remove: self.board_pieces[pr][pc] = 0
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for i in range(4):
                if axis_found[i]:
                    dr, dc = directions[i]
                    for sign in [1, -1]:
                        cr, cc = r, c
                        while 0 <= cr < self.board_size and 0 <= cc < self.board_size:
                            if self.board_pieces[cr][cc] == -player_who_moved: break
                            self.board_territory[cr][cc] = player_who_moved;
                            cr += sign * dr;
                            cc += sign * dc
        self.current_move_number += 1;
        self.current_player *= -1
        return True, line_centers, player_who_moved

    def check_game_end(self):
        if self.current_move_number >= self.max_total_moves:
            p1_score = sum(row.count(1) for row in self.board_territory)
            p2_score = sum(row.count(-1) for row in self.board_territory)
            return p1_score, p2_score, True
        return sum(row.count(1) for row in self.board_territory), sum(
            row.count(-1) for row in self.board_territory), False


class Particle:
    def __init__(self, x, y, color, life, angle, speed, size):
        self.x = x;
        self.y = y;
        self.color = color;
        self.life = life;
        self.max_life = life;
        self.size = size
        self.vx = math.cos(angle) * speed;
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx;
        self.y += self.vy;
        self.life -= 1
        return self.life > 0

    def draw(self, screen):
        if self.life > 0:
            alpha = int(255 * max(0, self.life / self.max_life))
            current_color = self.color + (alpha,)
            s = pygame.Surface((self.size, self.size), pygame.SRCALPHA);
            s.fill(current_color)
            screen.blit(s, (self.x - self.size / 2, self.y - self.size / 2))


class GameGUI:
    def __init__(self):
        pygame.init()
        self.screen_size = (960, 720)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("MyAIChess - 稳定最终版")
        self.font_path = "C:\Windows\Fonts\simhei.ttf"
        try:
            self.font_big = pygame.font.Font(self.font_path, 48);
            self.font_medium = pygame.font.Font(self.font_path, 32);
            self.font_small = pygame.font.Font(self.font_path, 22)
        except FileNotFoundError:
            print(f"警告：字体文件 {self.font_path} 未找到！");
            self.font_big = pygame.font.Font(None, 48);
            self.font_medium = pygame.font.Font(None, 32);
            self.font_small = pygame.font.Font(None, 22)
        self.clock = pygame.time.Clock()
        self.game = PythonGomoku(
            board_size=args['board_size'],
            num_rounds=args['num_rounds']
        )
        self.particles = []
        self.human_player = 1
        self.model_file = find_latest_model_file()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.colors = {'p1': (220, 50, 120), 'p2': (50, 200, 120), 'background': (25, 25, 40),
                       'grid_line': (40, 40, 60), 'territory1': (200, 0, 100, 100), 'territory2': (0, 200, 100, 100)}

        # ★★★ 核心修正 1：不再需要锁和共享状态，改用队列 ★★★
        self.ai_result_queue = queue.Queue()
        self.ai_is_thinking = False  # 这个标志现在只由主线程管理

    def draw_placeholder_piece(self, surface, player, center, radius, color=None):
        if color is None: color = self.colors['p1'] if player == 1 else self.colors['p2']
        if player == 1:
            pygame.draw.circle(surface, color, center, radius)
        elif player == -1:
            points = [(center[0], center[1] - radius), (center[0] + radius, center[1]), (center[0], center[1] + radius),
                      (center[0] - radius, center[1]), ]
            pygame.draw.polygon(surface, color, points)

    def create_effect(self, position, count, color, effect_type='burst'):
        for _ in range(count):
            if effect_type == 'burst':
                angle = random.uniform(0, 2 * math.pi);
                speed = random.uniform(1, 4);
                size = random.randint(3, 8)
            elif effect_type == 'line':
                angle = random.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2]) + random.uniform(-0.2, 0.2);
                speed = random.uniform(2, 5);
                size = random.randint(5, 12)
            life = random.randint(30, 60)
            self.particles.append(Particle(position[0], position[1], color, life, angle, speed, size))

    def draw_game_scene(self):
        # ... 此函数内容保持不变 ...
        CELL_SIZE = 60
        BOARD_START_X = (self.screen_size[0] - self.game.board_size * CELL_SIZE) // 2
        BOARD_START_Y = (self.screen_size[1] - self.game.board_size * CELL_SIZE) // 2
        self.screen.fill(self.colors['background'])
        for r in range(self.game.board_size):
            for c in range(self.game.board_size):
                rect = pygame.Rect(BOARD_START_X + c * CELL_SIZE, BOARD_START_Y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, self.colors['grid_line'], rect, 2)
                territory = self.game.board_territory[r][c]
                if territory != 0:
                    t_color = self.colors['territory1'] if territory == 1 else self.colors['territory2']
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA);
                    s.fill(t_color);
                    self.screen.blit(s, rect.topleft)
                    self.draw_placeholder_piece(self.screen, territory, rect.center, CELL_SIZE // 4,
                                                self.colors['background'])
                piece = self.game.board_pieces[r][c]
                if piece != 0: self.draw_placeholder_piece(self.screen, piece, rect.center, CELL_SIZE // 2 - 8)
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles: p.draw(self.screen)
        p1_score, p2_score, _ = self.game.check_game_end()
        rem_moves = self.game.max_total_moves - self.game.current_move_number
        top_text = self.font_medium.render(f"剩余回合: {rem_moves}", True, (200, 200, 200))
        self.screen.blit(top_text, (self.screen_size[0] // 2 - top_text.get_width() // 2, 20))
        self.draw_placeholder_piece(self.screen, 1, (80, 100), 40);
        p1_text = self.font_big.render(f"{p1_score}", True, self.colors['p1']);
        self.screen.blit(p1_text, (80, 200))
        self.draw_placeholder_piece(self.screen, -1, (self.screen_size[0] - 80, 100), 40);
        p2_text = self.font_big.render(f"{p2_score}", True, self.colors['p2']);
        self.screen.blit(p2_text, (self.screen_size[0] - 80 - p2_text.get_width(), 200))

        # file: play_pixel_art.py

        # ... (其他代码保持不变) ...

        # ★★★ 核心修正 2：AI工作函数现在只负责计算和把结果“投递”到邮箱 ★★★

    def _ai_worker_func(self, board_pieces, board_territory, current_player, current_move_number):

        # ====================== [核心修正] ======================
        # 从全局 config.py 的 args 字典创建一份完整的参数副本
        # 以确保人机对战的AI行为与训练时完全一致
        ai_args = args.copy()

        # vvvvvv 【新增的修复代码】 vvvvvv
        # 从 coach.py 复制过来的参数计算逻辑，确保C++引擎获得必要的num_channels
        history_channels = (ai_args.get('history_steps', 0) + 1) * 4
        meta_channels = 4
        total_channels = history_channels + meta_channels
        ai_args['num_channels'] = total_channels
        # ^^^^^^ 【新增的修复代码】 ^^^^^^

        # 你可以根据需要覆盖特定参数，例如，在对战时减少搜索次数
        # ai_args['num_searches'] = 400 # 示例：如果想在对战时降低AI强度

        # C++ 引擎需要的其他参数，如 board_size 和 max_total_moves，也应更新
        ai_args['board_size'] = self.game.board_size
        ai_args['max_total_moves'] = self.game.max_total_moves
        # ========================================================

        # vvvvvvvv 使用完整的 ai_args 字典 vvvvvvvv
        best_action = cpp_mcts_engine.find_best_action(board_pieces, board_territory, current_player,
                                                       current_move_number, self.model_file,
                                                       self.device.type == 'cuda',
                                                       ai_args)
        self.ai_result_queue.put(best_action)

    # ... (run 函数及后续代码保持不变) ...
    def run(self):
        CELL_SIZE = 60
        BOARD_START_X = (self.screen_size[0] - self.game.board_size * CELL_SIZE) // 2
        BOARD_START_Y = (self.screen_size[1] - self.game.board_size * CELL_SIZE) // 2
        running = True;
        game_over = False

        while running:
            # ★★★ 核心修正 3：重构主循环，使用队列通信 ★★★

            # 1. 检查邮箱，看AI是否已经把结果送回来了
            try:
                ai_action = self.ai_result_queue.get_nowait()
                # 如果能取到东西，说明AI思考完了
                valid, line_centers, player_who_moved = self.game.execute_move(ai_action)
                if valid:
                    r, c = ai_action // self.game.board_size, ai_action % self.game.board_size
                    pos = (
                        BOARD_START_X + c * CELL_SIZE + CELL_SIZE // 2, BOARD_START_Y + r * CELL_SIZE + CELL_SIZE // 2)
                    self.create_effect(pos, 30, (200, 200, 255), 'burst')
                    if line_centers:
                        line_color = self.colors['p1'] if player_who_moved == 1 else self.colors['p2']
                        for lr, lc in line_centers:
                            l_pos = (BOARD_START_X + lc * CELL_SIZE + CELL_SIZE // 2,
                                     BOARD_START_Y + lr * CELL_SIZE + CELL_SIZE // 2)
                            self.create_effect(l_pos, 50, line_color, 'line')
                self.ai_is_thinking = False  # AI执行完毕，标记为空闲
            except queue.Empty:
                # 邮箱是空的，这是正常情况，什么都不用做
                pass

            # 2. 处理用户输入
            is_human_turn = (self.game.current_player == self.human_player and not game_over)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if is_human_turn and not self.ai_is_thinking and event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    c = (x - BOARD_START_X) // CELL_SIZE;
                    r = (y - BOARD_START_Y) // CELL_SIZE
                    if 0 <= r < self.game.board_size and 0 <= c < self.game.board_size:
                        action = r * self.game.board_size + c
                        valid, line_centers, player_who_moved = self.game.execute_move(action)
                        if valid:
                            pos = (BOARD_START_X + c * CELL_SIZE + CELL_SIZE // 2,
                                   BOARD_START_Y + r * CELL_SIZE + CELL_SIZE // 2)
                            self.create_effect(pos, 30, (255, 255, 200), 'burst')
                            if line_centers:
                                line_color = self.colors['p1'] if player_who_moved == 1 else self.colors['p2']
                                for lr, lc in line_centers:
                                    l_pos = (BOARD_START_X + lc * CELL_SIZE + CELL_SIZE // 2,
                                             BOARD_START_Y + lr * CELL_SIZE + CELL_SIZE // 2)
                                    self.create_effect(l_pos, 50, line_color, 'line')

            # 3. 检查是否需要启动AI线程
            is_ai_turn_to_start = (
                    self.game.current_player != self.human_player and not game_over and not self.ai_is_thinking)
            if is_ai_turn_to_start:
                self.ai_is_thinking = True
                thread_args = (copy.deepcopy(self.game.board_pieces), copy.deepcopy(self.game.board_territory),
                               self.game.current_player, self.game.current_move_number)
                ai_thread = threading.Thread(target=self._ai_worker_func, args=thread_args, daemon=True)
                ai_thread.start()

            # 4. 绘制所有内容
            self.draw_game_scene()
            if self.ai_is_thinking:
                thinking_text = self.font_medium.render("AI 正在思考...", True, (255, 255, 100))
                self.screen.blit(thinking_text,
                                 (self.screen_size[0] - thinking_text.get_width() - 20, self.screen_size[1] - 50))

            if not game_over:
                p1_score, p2_score, is_ended = self.game.check_game_end()
                if is_ended:
                    game_over = True
                    winner_text = "平局"
                    if p1_score > p2_score:
                        winner_text = "粉方 获胜!"
                    elif p2_score > p1_score:
                        winner_text = "绿方 获胜!"

            if game_over:
                s = pygame.Surface(self.screen_size, pygame.SRCALPHA);
                s.fill((0, 0, 0, 180));
                self.screen.blit(s, (0, 0))
                end_text = self.font_big.render(winner_text, True, (255, 255, 100))
                self.screen.blit(end_text, (self.screen_size[0] // 2 - end_text.get_width() // 2,
                                            self.screen_size[1] // 2 - end_text.get_height() // 2))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    game_gui = GameGUI()
    game_gui.run()
