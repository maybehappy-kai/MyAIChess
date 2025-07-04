# file: play_pixel_art.py (最终修正版 - v3)
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
import queue
import pickle
import numpy as np


def find_latest_model_file():
    path = "."
    max_epoch = -1
    latest_file = None
    pattern = re.compile(r"model_(\d+).*\.pt")
    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch: max_epoch = epoch; latest_file = f
    return latest_file


class PythonGomoku:
    def __init__(self, board_size=9, num_rounds=25):
        self.board_size = board_size
        self.max_total_moves = num_rounds * 2
        self.last_move = None
        self.reset()

    def get_state_for_training(self):
        # 此函数用于为训练生成状态，与之前版本保持一致
        num_channels = args.get('num_channels', 20)
        state = np.zeros((num_channels, self.board_size, self.board_size))
        # 视角是相对于当前玩家的，p1是当前玩家，p2是对手
        p1 = self.current_player
        p2 = -self.current_player
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board_pieces[r][c] == p1: state[0, r, c] = 1
                if self.board_pieces[r][c] == p2: state[1, r, c] = 1
                if self.board_territory[r][c] == p1: state[2, r, c] = 1
                if self.board_territory[r][c] == p2: state[3, r, c] = 1
        # 元数据通道
        state[num_channels - 4].fill(1 if self.current_player == 1 else 0)
        return state.flatten().tolist()

    def reset(self):
        self.board_pieces = [[0] * self.board_size for _ in range(self.board_size)]
        self.board_territory = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1  # 黑棋(1)总是先手
        self.current_move_number = 0
        self.last_move = None

    def get_valid_moves(self):
        valid_moves = [False] * (self.board_size * self.board_size)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board_pieces[r][c] == 0 and self.board_territory[r][c] != -self.current_player:
                    valid_moves[r * self.board_size + c] = True
        return valid_moves

    def execute_move(self, action):
        if not self.get_valid_moves()[action]: return False, [], 0
        r, c = action // self.board_size, action % self.board_size
        player_who_moved = self.current_player
        self.board_pieces[r][c] = player_who_moved
        self.last_move = (r, c)

        # ==================== BUG修复区域 ====================
        # line_centers 的逻辑有误，现改为直接返回所有参与三连的棋子坐标
        pieces_to_remove = set()
        # =====================================================

        axis_found = [False] * 4
        combinations = [[(0, -2), (0, -1), (0, 0)], [(0, -1), (0, 0), (0, 1)], [(0, 0), (0, 1), (0, 2)],
                        [(-2, 0), (-1, 0), (0, 0)], [(-1, 0), (0, 0), (1, 0)], [(0, 0), (1, 0), (2, 0)],
                        [(-2, -2), (-1, -1), (0, 0)], [(-1, -1), (0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)],
                        [(2, -2), (1, -1), (0, 0)], [(1, -1), (0, 0), (-1, 1)], [(0, 0), (-1, 1), (-2, 2)]]

        for i, combo in enumerate(combinations):
            points = [(r + dr, c + dc) for dr, dc in combo]
            if all(0 <= pr < self.board_size and 0 <= pc < self.board_size and self.board_pieces[pr][
                pc] == player_who_moved for pr, pc in points):
                pieces_to_remove.update(points)
                axis_found[i // 3] = True

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
                            self.board_territory[cr][cc] = player_who_moved
                            cr += sign * dr
                            cc += sign * dc

        self.current_move_number += 1
        self.current_player *= -1

        # ==================== BUG修复区域 ====================
        # 返回所有参与形成三连的棋子坐标列表，而不再是之前的 line_centers
        return True, list(pieces_to_remove), player_who_moved
        # =====================================================

    def check_game_end(self):
        if self.current_move_number >= self.max_total_moves or not any(self.get_valid_moves()):
            p1_score = sum(row.count(1) for row in self.board_territory)
            p2_score = sum(row.count(-1) for row in self.board_territory)
            return p1_score, p2_score, True
        return sum(row.count(1) for row in self.board_territory), sum(
            row.count(-1) for row in self.board_territory), False


class Particle:
    def __init__(self, x, y, color, life, angle, speed, size):
        self.x, self.y, self.color, self.life, self.max_life, self.size = x, y, color, life, life, size
        self.vx, self.vy = math.cos(angle) * speed, math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, screen):
        if self.life > 0:
            alpha = int(255 * max(0, self.life / self.max_life))
            s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            s.fill(self.color + (alpha,))
            screen.blit(s, (self.x - self.size / 2, self.y - self.size / 2))


class GameGUI:
    def __init__(self):
        pygame.init()
        self.screen_size = (960, 720)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("MyAIChess - 专家数据采集中")
        try:
            self.font_big = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 48)
            self.font_medium = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 32)
            self.font_small = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 22)
        except FileNotFoundError:
            self.font_big, self.font_medium, self.font_small = pygame.font.Font(None, 48), pygame.font.Font(None,
                                                                                                            32), pygame.font.Font(
                None, 24)

        self.clock = pygame.time.Clock()
        self.game = PythonGomoku(board_size=args['board_size'], num_rounds=args['num_rounds'])
        self.particles = []

        self.game_state = "CHOOSING_SIDE"
        self.human_player = None
        self.game_was_completed = False
        self.combo_highlights = []

        self.model_file = find_latest_model_file()
        if self.model_file:
            print(f"AI 使用模型: {self.model_file}")
        else:
            print("警告: 未找到任何.pt模型, AI将无法行动！")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.colors = {'p1': (220, 50, 120), 'p2': (50, 200, 120), 'background': (25, 25, 40),
                       'grid_line': (40, 40, 60), 'territory1': (200, 0, 100, 100), 'territory2': (0, 200, 100, 100),
                       # ==================== BUG修复区域：颜色已改为白色 ====================
                       'highlight_last_move': (255, 255, 255, 100),  # 亮白色，半透明
                       'highlight_combo': (255, 255, 255, 60)  # 暗白色，更透明
                       # =====================================================================
                       }
        self.ai_result_queue, self.ai_is_thinking = queue.Queue(), False
        self.game_history = []

        button_width, button_height, button_margin = 180, 60, 20
        self.restart_button_rect = pygame.Rect(self.screen_size[0] - button_width - button_margin,
                                               self.screen_size[1] - button_height - button_margin, button_width,
                                               button_height)
        self.exit_button_rect = pygame.Rect(self.screen_size[0] - (button_width + button_margin) * 2,
                                            self.screen_size[1] - button_height - button_margin, button_width,
                                            button_height)
        center_x, center_y = self.screen_size[0] // 2, self.screen_size[1] // 2
        self.play_black_button_rect = pygame.Rect(center_x - button_width - button_margin,
                                                  center_y - button_height // 2, button_width, button_height)
        self.play_white_button_rect = pygame.Rect(center_x + button_margin, center_y - button_height // 2, button_width,
                                                  button_height)
        self.button_color = (0, 100, 200)
        self.button_text_color = (255, 255, 255)

    def draw_placeholder_piece(self, surface, player, center, radius, color=None):
        color = color if color else (self.colors['p1'] if player == 1 else self.colors['p2'])
        if player == 1:
            pygame.draw.circle(surface, color, center, radius)
        else:
            pygame.draw.polygon(surface, color, [(center[0], center[1] - radius), (center[0] + radius, center[1]),
                                                 (center[0], center[1] + radius), (center[0] - radius, center[1])])

    def create_effect(self, position, count, color, effect_type='burst'):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi) if effect_type == 'burst' else random.choice(
                [0, math.pi / 2, math.pi, 3 * math.pi / 2]) + random.uniform(-0.2, 0.2)
            speed, size = (random.uniform(1, 4), random.randint(3, 8)) if effect_type == 'burst' else (
            random.uniform(2, 5), random.randint(5, 12))
            self.particles.append(Particle(position[0], position[1], color, random.randint(30, 60), angle, speed, size))

    def draw_game_scene(self):
        CELL_SIZE, BOARD_START_X, BOARD_START_Y = 60, (self.screen_size[0] - self.game.board_size * 60) // 2, (
                    self.screen_size[1] - self.game.board_size * 60) // 2
        self.screen.fill(self.colors['background'])

        for r in range(self.game.board_size):
            for c in range(self.game.board_size):
                rect = pygame.Rect(BOARD_START_X + c * CELL_SIZE, BOARD_START_Y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, self.colors['grid_line'], rect, 2)
                if self.game.board_territory[r][c] != 0:
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    s.fill(self.colors['territory1'] if self.game.board_territory[r][c] == 1 else self.colors[
                        'territory2'])
                    self.screen.blit(s, rect.topleft)
                    self.draw_placeholder_piece(self.screen, self.game.board_territory[r][c], rect.center,
                                                CELL_SIZE // 4, self.colors['background'])
                if self.game.board_pieces[r][c] != 0:
                    self.draw_placeholder_piece(self.screen, self.game.board_pieces[r][c], rect.center,
                                                CELL_SIZE // 2 - 8)

        if self.combo_highlights:
            for r_combo, c_combo in self.combo_highlights:
                if (r_combo, c_combo) != self.game.last_move:
                    highlight_rect = pygame.Rect(BOARD_START_X + c_combo * CELL_SIZE,
                                                 BOARD_START_Y + r_combo * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    s.fill(self.colors['highlight_combo'])
                    self.screen.blit(s, highlight_rect.topleft)

        if self.game.last_move:
            lr, lc = self.game.last_move
            highlight_rect = pygame.Rect(BOARD_START_X + lc * CELL_SIZE, BOARD_START_Y + lr * CELL_SIZE, CELL_SIZE,
                                         CELL_SIZE)
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill(self.colors['highlight_last_move'])
            self.screen.blit(s, highlight_rect.topleft)

        self.particles = [p for p in self.particles if p.update()]
        [p.draw(self.screen) for p in self.particles]
        p1s, p2s, _ = self.game.check_game_end()
        rem_moves = self.game.max_total_moves - self.game.current_move_number
        top_text = self.font_medium.render(f"剩余回合: {rem_moves}", True, (200, 200, 200))
        self.screen.blit(top_text, (self.screen_size[0] // 2 - top_text.get_width() // 2, 20))
        self.draw_placeholder_piece(self.screen, 1, (80, 100), 40)
        p1t = self.font_big.render(f"{p1s}", True, self.colors['p1'])
        self.screen.blit(p1t, (80, 200))
        self.draw_placeholder_piece(self.screen, -1, (self.screen_size[0] - 80, 100), 40)
        p2t = self.font_big.render(f"{p2s}", True, self.colors['p2'])
        self.screen.blit(p2t, (self.screen_size[0] - 80 - p2t.get_width(), 200))
        pygame.draw.rect(self.screen, self.button_color, self.restart_button_rect, border_radius=10)
        restart_text = self.font_small.render("重开本局", True, self.button_text_color)
        self.screen.blit(restart_text, restart_text.get_rect(center=self.restart_button_rect.center))
        pygame.draw.rect(self.screen, self.button_color, self.exit_button_rect, border_radius=10)
        exit_text = self.font_small.render("退出程序", True, self.button_text_color)
        self.screen.blit(exit_text, exit_text.get_rect(center=self.exit_button_rect.center))

    def _ai_worker_func(self, board_pieces, board_territory, current_player, current_move_number):
        ai_args = args.copy()
        ai_args['num_channels'] = (args.get('history_steps', 0) + 1) * 4 + 4
        ai_args['board_size'], ai_args['max_total_moves'] = self.game.board_size, self.game.max_total_moves
        action = cpp_mcts_engine.find_best_action(board_pieces, board_territory, current_player, current_move_number,
                                                  self.model_file, self.device.type == 'cuda', ai_args)
        if self.ai_is_thinking: self.ai_result_queue.put(action)

    def save_game_data(self, game_result):
        if not self.game_history: return
        final_data = []
        current_value = game_result
        for state, policy, _ in reversed(self.game_history):
            final_data.append((state, policy, current_value, self.human_player))
            current_value *= -1
        final_data.reverse()
        try:
            expert_data_file = 'human_games.pkl'
            if os.path.exists(expert_data_file):
                with open(expert_data_file, 'rb') as f:
                    existing_data = pickle.load(f)
            else:
                existing_data = []
            existing_data.extend(final_data)
            total_human_first_data = sum(1 for (*_, human_side) in existing_data if human_side == 1)
            total_human_second_data = sum(1 for (*_, human_side) in existing_data if human_side == -1)
            max_size = args.get('expert_data_max_size', 2000)
            recent_data = existing_data[-max_size:]
            recent_human_first = sum(1 for (*_, human_side) in recent_data if human_side == 1)
            recent_human_second = sum(1 for (*_, human_side) in recent_data if human_side == -1)
            with open(expert_data_file, 'wb') as f:
                pickle.dump(existing_data, f)
            print("\n" + "=" * 20 + " 数据收集报告 " + "=" * 20)
            print(f"本局收集到 {len(final_data)} 步棋。")
            print(f"总计专家数据量: {len(existing_data)} 条。")
            print(f"  - 人类先手 (执黑) 数据: {total_human_first_data} 条")
            print(f"  - 人类后手 (执白) 数据: {total_human_second_data} 条")
            print(f"最近 {max_size} 条数据中:")
            print(f"  - 人类先手: {recent_human_first} 条 | 人类后手: {recent_human_second} 条")
            print("=" * 58)
        except Exception as e:
            print(f"[错误] 保存专家数据失败: {e}")
        self.game_history.clear()
        self.game_was_completed = True

    def reset_game_and_ai_state(self):
        if self.game_was_completed:
            print("\n--- 先前对局已收集，开始新的一局 ---")
        else:
            print("\n--- 新的一局 (当前对局数据已放弃) ---")
        self.game_was_completed = False

        self.game.reset()
        self.game_history.clear()
        self.combo_highlights = []

        self.ai_is_thinking = False
        while not self.ai_result_queue.empty():
            try:
                self.ai_result_queue.get_nowait()
            except queue.Empty:
                break
        self.game_state = "CHOOSING_SIDE"

    def run(self):
        CELL_SIZE, BOARD_START_X, BOARD_START_Y = 60, (self.screen_size[0] - self.game.board_size * 60) // 2, (
                    self.screen_size[1] - self.game.board_size * 60) // 2
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.restart_button_rect.collidepoint(event.pos):
                        self.reset_game_and_ai_state()
                        continue
                    if self.exit_button_rect.collidepoint(event.pos):
                        running = False
                        continue

                    if self.game_state == "CHOOSING_SIDE":
                        if self.play_black_button_rect.collidepoint(event.pos):
                            self.human_player = 1;
                            self.game_state = "PLAYING";
                            print("您选择执黑先行。")
                        elif self.play_white_button_rect.collidepoint(event.pos):
                            self.human_player = -1;
                            self.game_state = "PLAYING";
                            print("您选择执白后行。")

                    elif self.game_state == "PLAYING":
                        is_human_turn = self.game.current_player == self.human_player
                        if is_human_turn and not self.ai_is_thinking:
                            x, y = event.pos
                            c, r = (x - BOARD_START_X) // CELL_SIZE, (y - BOARD_START_Y) // CELL_SIZE
                            if 0 <= r < self.game.board_size and 0 <= c < self.game.board_size:
                                action = r * self.game.board_size + c
                                if self.game.get_valid_moves()[action]:
                                    state_before = self.game.get_state_for_training()
                                    alpha, action_size = args.get('label_smoothing_alpha',
                                                                  0.03), self.game.board_size ** 2
                                    policy = [alpha / (action_size - 1)] * action_size
                                    policy[action] = 1.0 - alpha
                                    self.game_history.append((state_before, policy, None))

                                    # ==================== BUG修复区域：捕获正确的combo_pieces ====================
                                    valid, combo_pieces, _ = self.game.execute_move(action)
                                    self.combo_highlights = combo_pieces
                                    # =========================================================================

                                    if valid:
                                        pos = (BOARD_START_X + c * CELL_SIZE + CELL_SIZE // 2,
                                               BOARD_START_Y + r * CELL_SIZE + CELL_SIZE // 2)
                                        self.create_effect(pos, 30, (255, 255, 200), 'burst')

            if self.game_state == "PLAYING":
                if self.ai_is_thinking:
                    try:
                        ai_action = self.ai_result_queue.get_nowait()
                        state_before = self.game.get_state_for_training()
                        policy_ph = [0.0] * (self.game.board_size ** 2)
                        self.game_history.append((state_before, policy_ph, None))

                        # ==================== BUG修复区域：捕获正确的combo_pieces ====================
                        valid, combo_pieces, _ = self.game.execute_move(ai_action)
                        self.combo_highlights = combo_pieces
                        # =========================================================================

                        if valid:
                            r, c = ai_action // self.game.board_size, ai_action % self.game.board_size
                            pos = (BOARD_START_X + c * CELL_SIZE + CELL_SIZE // 2,
                                   BOARD_START_Y + r * CELL_SIZE + CELL_SIZE // 2)
                            self.create_effect(pos, 30, (200, 200, 255), 'burst')
                        self.ai_is_thinking = False
                    except queue.Empty:
                        pass

                p1s, p2s, is_ended = self.game.check_game_end()
                if is_ended:
                    self.game_state = "GAME_OVER"
                    result = 0.001 if p1s == p2s else (1.0 if p1s > p2s else -1.0)
                    self.save_game_data(result)
                else:
                    is_ai_turn = self.game.current_player != self.human_player
                    if is_ai_turn and not self.ai_is_thinking and self.model_file:
                        self.ai_is_thinking = True
                        t_args = (copy.deepcopy(self.game.board_pieces), copy.deepcopy(self.game.board_territory),
                                  self.game.current_player, self.game.current_move_number)
                        threading.Thread(target=self._ai_worker_func, args=t_args, daemon=True).start()

            self.screen.fill(self.colors['background'])
            if self.game_state == "CHOOSING_SIDE":
                title_text = self.font_big.render("请选择您的阵营", True, self.button_text_color)
                self.screen.blit(title_text,
                                 title_text.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] // 2 - 100)))
                pygame.draw.rect(self.screen, self.button_color, self.play_black_button_rect, border_radius=10)
                black_text = self.font_medium.render("执黑先行", True, self.button_text_color)
                self.screen.blit(black_text, black_text.get_rect(center=self.play_black_button_rect.center))
                pygame.draw.rect(self.screen, self.button_color, self.play_white_button_rect, border_radius=10)
                white_text = self.font_medium.render("执白后行", True, self.button_text_color)
                self.screen.blit(white_text, white_text.get_rect(center=self.play_white_button_rect.center))
            else:
                self.draw_game_scene()
                if self.ai_is_thinking:
                    thinking_text = self.font_medium.render("AI 正在思考...", True, (255, 255, 100))
                    self.screen.blit(thinking_text, (20, self.screen_size[1] - 50))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    args['num_channels'] = (args.get('history_steps', 0) + 1) * 4 + 4
    GameGUI().run()