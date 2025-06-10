# play_game.py
import pygame, sys, os, random, math, copy, numpy as np, torch, threading
#from game import Game as AIGame
from neural_net import ExtendedConnectNet
from config import args


# --- 为本文件内置一个专用的、简单的MCTS类，与训练代码完全解耦 ---
class Node:
    def __init__(self, game, args, parent=None, action_taken=None, prior=0):
        self.game = game; self.args = args; self.parent = parent; self.action_taken = action_taken; self.prior = prior; self.children = []; self.visit_count = 0; self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None;
        best_ucb = -np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb: best_ucb = ucb; best_child = child
        return best_child

    def get_ucb(self, child):
        q_value = -child.value_sum / child.visit_count if child.visit_count > 0 else 0
        u_value = self.args['C'] * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count);
        return q_value + u_value

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_game = self.game.copy();
                child_game.execute_move(action)
                child = Node(child_game, self.args, parent=self, action_taken=action, prior=prob);
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value;
        self.visit_count += 1
        if self.parent is not None: self.parent.backpropagate(-value)


class MCTS_for_play:  # 使用不同的类名避免任何潜在冲突
    def __init__(self, game, args, model):
        self.game = game; self.args = args; self.model = model

    @torch.no_grad()
    def search(self, game, add_noise=False):
        root = Node(game, self.args);
        model_device = next(self.model.parameters()).device
        state_tensor = torch.tensor(root.game.get_state()).unsqueeze(0).float().to(model_device)
        policy, value = self.model(state_tensor);
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy();
        value = value.item()
        valid_moves = root.game.get_valid_moves();
        policy *= valid_moves
        if np.sum(policy) > 0: policy /= np.sum(policy)
        root.expand(policy)
        for _ in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded(): node = node.select()
            value, is_terminal = node.game.get_game_ended()
            if not is_terminal:
                state_tensor = torch.tensor(node.game.get_state()).unsqueeze(0).float().to(model_device);
                policy, value_tensor = self.model(state_tensor)
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy();
                valid_moves = node.game.get_valid_moves();
                policy *= valid_moves
                if np.sum(policy) == 0 and np.sum(valid_moves) > 0:
                    policy = valid_moves / np.sum(valid_moves)
                elif np.sum(policy) > 0:
                    policy /= np.sum(policy)
                value = value_tensor.item();
                node.expand(policy)
            node.backpropagate(value)
        action_probs = np.zeros(game.BOARD_SIZE ** 2)
        for child in root.children: action_probs[child.action_taken] = child.visit_count
        if np.sum(action_probs) > 0: action_probs /= np.sum(action_probs)
        return action_probs


# --- 常量、颜色、字体、窗口初始化 ---
BOARD_SIZE = 9;
CELL_SIZE = 60;
PADDING = 20;
INFO_AREA_HEIGHT = 180
COLOR_BACKGROUND = pygame.Color(15, 20, 35);
COLOR_BOARD_BASE = pygame.Color(25, 30, 50);
COLOR_BOARD_GLYPH = pygame.Color(40, 50, 75);
COLOR_GRID_PULSE_LIGHT = pygame.Color(80, 100, 140);
COLOR_GRID_PULSE_DARK = pygame.Color(50, 65, 95);
COLOR_GROOVE_SHADOW = pygame.Color(10, 15, 28);
COLOR_HOVER_HIGHLIGHT = pygame.Color(200, 255, 255);
COLOR_PLAYER_B = pygame.Color(0, 255, 255);
COLOR_PLAYER_W = pygame.Color(255, 0, 255);
COLOR_TERRITORY_B_BG = (COLOR_PLAYER_B.r, COLOR_PLAYER_B.g, COLOR_PLAYER_B.b, 100);
COLOR_TERRITORY_W_BG = (COLOR_PLAYER_W.r, COLOR_PLAYER_W.g, COLOR_PLAYER_W.b, 100);
COLOR_PARTICLE_B = COLOR_PLAYER_B;
COLOR_PARTICLE_W = COLOR_PLAYER_W;
COLOR_TERRITORY_HIGHLIGHT = pygame.Color("white");
COLOR_UI_BG = pygame.Color(30, 40, 60);
COLOR_UI_BORDER = pygame.Color(15, 20, 35);
COLOR_UI_TEXT = pygame.Color(200, 220, 255);
COLOR_UI_HIGHLIGHT = pygame.Color(255, 255, 150);
COLOR_UI_ERROR = pygame.Color(255, 80, 120);
COLOR_BUTTON_BASE = pygame.Color(60, 70, 100);
COLOR_BUTTON_HOVER = pygame.Color(90, 110, 140);
COLOR_BUTTON_TEXT = pygame.Color(220, 230, 255)
EMPTY_SLOT = None;
PLAYER_BLACK = 'B';
PLAYER_WHITE = 'W'
GAME_MODE_DUAL = "双人游戏";
GAME_MODE_AI = "单人游戏";
PLAYER_HUMAN = "人类";
PLAYER_AI = "AI"
pygame.init()
try:
    game_font = pygame.font.SysFont('Microsoft YaHei', 20, bold=True);
    small_game_font = pygame.font.SysFont('Microsoft YaHei', 16, bold=True);
    title_font = pygame.font.SysFont('Microsoft YaHei', 48, bold=True);
    menu_button_font = pygame.font.SysFont('Microsoft YaHei', 24, bold=True);
    game_button_font = pygame.font.SysFont('Microsoft YaHei', 18, bold=True)
except pygame.error:
    game_font = pygame.font.Font(None, 28);
    small_game_font = pygame.font.Font(None, 22);
    title_font = pygame.font.Font(None, 60);
    menu_button_font = pygame.font.Font(None, 32);
    game_button_font = pygame.font.Font(None, 24)
screen_width = BOARD_SIZE * CELL_SIZE + 2 * PADDING;
screen_height = BOARD_SIZE * CELL_SIZE + 2 * PADDING + INFO_AREA_HEIGHT
screen = pygame.display.set_mode((screen_width, screen_height));
pygame.display.set_caption("扩展连线棋")

# --- 全局状态变量和AI模型 ---
MODEL_FILE = 'model_2.pth';
AI_MODEL = None;
game_mode = None;
player_roles = {PLAYER_BLACK: None, PLAYER_WHITE: None};
board_pieces = [];
board_territory = []
current_player = PLAYER_BLACK;
game_over = False;
message = "";
current_move_number = 0;
max_total_moves = 0
num_rounds_per_player_default = 25;
active_particle_spawners = [];
history = [];
particles = []
ai_is_thinking = False;
ai_move_result = None


# --- 所有函数定义 ---
class Button:
    def __init__(self, rect, text, action, font=game_button_font): self.rect = pygame.Rect(
        rect); self.text = text; self.action = action; self.font = font

    def draw(self, surface, mouse_pos):
        is_hovered = self.rect.collidepoint(mouse_pos);
        color = COLOR_BUTTON_HOVER if is_hovered else COLOR_BUTTON_BASE;
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, COLOR_UI_BORDER, self.rect, 2, border_radius=5);
        text_surf = self.font.render(self.text, True, COLOR_BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center);
        surface.blit(text_surf, text_rect)

    def check_click(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(
            event.pos): return self.action
        return None


class Particle:
    def __init__(self, x, y, color, speed, direction, lifespan, size,
                 gravity=0.0): self.x, self.y = x, y; self.color, self.lifespan, self.size, self.gravity = color, lifespan, size, gravity; self.max_lifespan = lifespan; self.vx = speed * math.cos(
        direction); self.vy = speed * math.sin(direction)

    def update(self): self.x += self.vx; self.y += self.vy; self.vy += self.gravity; self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, 255 * (self.lifespan / self.max_lifespan));
            s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            s.fill((self.color.r, self.color.g, self.color.b, alpha));
            surface.blit(s, (self.x - self.size / 2, self.y - self.size / 2), special_flags=pygame.BLEND_RGBA_ADD)


def create_particle_burst(x, y, p_color, count=25, gravity=0.05):
    for _ in range(count): particles.append(
        Particle(x, y, p_color, random.uniform(0.5, 2.5), random.uniform(0, 2 * math.pi), random.randint(30, 60),
                 random.randint(2, 4), gravity))


def create_continuous_particle(x, y, p_color):
    angle = random.uniform(0, 2 * math.pi);
    speed = random.uniform(0.2, 0.5);
    lifespan = random.randint(40, 80);
    size = random.randint(2, 3)
    particles.append(Particle(x, y, p_color, speed, angle, lifespan, size, gravity=0))


def load_ai_model():
    global AI_MODEL
    if not os.path.exists(MODEL_FILE): print(f"错误：找不到AI模型文件 {MODEL_FILE}！"); return
    try:
        print("正在加载AI模型...");
        device = torch.device("cpu")
        model = ExtendedConnectNet(board_size=BOARD_SIZE, num_res_blocks=args['num_res_blocks'],
                                   num_hidden=args['num_hidden'])
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device));
        model.to(device);
        model.eval()
        AI_MODEL = model;
        print(f"AI模型加载成功!")
    except Exception as e:
        print(f"AI模型加载失败: {e}"); AI_MODEL = None


def get_state_snapshot(): return {'pieces': copy.deepcopy(board_pieces), 'territory': copy.deepcopy(board_territory),
                                  'player': current_player, 'move_num': current_move_number, 'message': message,
                                  'game_over': game_over}


def restore_state_from_snapshot(snapshot):
    global board_pieces, board_territory, current_player, current_move_number, message, game_over
    board_pieces = snapshot['pieces'];
    board_territory = snapshot['territory'];
    current_player = snapshot['player']
    current_move_number = snapshot['move_num'];
    message = snapshot['message'];
    game_over = snapshot['game_over'];
    active_particle_spawners.clear()


def undo_last_move():
    if not game_over and len(history) > 1:
        if game_mode == GAME_MODE_AI and player_roles[current_player] == PLAYER_AI:
            # 如果是AI的回合，意味着玩家刚下完，所以要撤回两步
            if len(history) > 2: history.pop(); history.pop(); restore_state_from_snapshot(history[-1])
        else:  # 双人模式，或轮到人类下棋时
            history.pop();
            restore_state_from_snapshot(history[-1])


def initialize_game_state(num_player_rounds):
    global board_pieces, board_territory, current_player, game_over, message, current_move_number, max_total_moves, particles, active_particle_spawners, history
    particles, active_particle_spawners, history = [], [], [];
    board_pieces = [[EMPTY_SLOT for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)];
    board_territory = [[EMPTY_SLOT for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = PLAYER_BLACK;
    game_over = False
    if game_mode == GAME_MODE_AI:
        message = "您执青方, 请下棋"
    else:
        message = "青方先行"
    current_move_number = 0;
    max_total_moves = num_player_rounds * 2;
    history.append(get_state_snapshot())


def place_piece(r, c, player):
    board_pieces[r][c] = player;
    center_x = c * CELL_SIZE + PADDING + CELL_SIZE // 2;
    center_y = r * CELL_SIZE + PADDING + CELL_SIZE // 2
    p_color = COLOR_PLAYER_B if player == PLAYER_BLACK else COLOR_PLAYER_W
    create_particle_burst(center_x, center_y, p_color, gravity=0);
    active_particle_spawners.append({'pos': (r, c), 'color': p_color})


def check_and_process_lines(player):
    global message;
    opponent = PLAYER_WHITE if player == PLAYER_BLACK else PLAYER_BLACK;
    lines_found_details = [];
    unique_lines_coords_sets = set()
    check_directions = [(0, 1), (1, 0), (1, 1), (1, -1)];
    has_line = False
    for r_start in range(BOARD_SIZE):
        for c_start in range(BOARD_SIZE):
            if board_pieces[r_start][c_start] == player:
                for dr, dc in check_directions:
                    p1 = (r_start, c_start);
                    p2_r, p2_c = r_start + dr, c_start + dc;
                    p3_r, p3_c = r_start + 2 * dr, c_start + 2 * dc
                    if not (
                            0 <= p2_r < BOARD_SIZE and 0 <= p2_c < BOARD_SIZE and 0 <= p3_r < BOARD_SIZE and 0 <= p3_c < BOARD_SIZE): continue
                    if board_pieces[p2_r][p2_c] == player and board_pieces[p3_r][p3_c] == player:
                        line_set = frozenset({p1, (p2_r, p2_c), (p3_r, p3_c)})
                        if line_set not in unique_lines_coords_sets: unique_lines_coords_sets.add(
                            line_set); lines_found_details.append(
                            {'coords_list': list(line_set), 'line_dr': dr, 'line_dc': dc})
    if not lines_found_details: return False
    has_line = True;
    pieces_to_remove = set();
    occupation_updates = {}
    for line in lines_found_details:
        for piece_coord in line['coords_list']: pieces_to_remove.add(piece_coord)
        for r_anchor, c_anchor in line['coords_list']:
            occupation_updates[(r_anchor, c_anchor)] = player;
            cr, cc = r_anchor + line['line_dr'], c_anchor + line['line_dc']
            while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                if board_pieces[cr][cc] == opponent: break
                occupation_updates[(cr, cc)] = player;
                cr, cc = cr + line['line_dr'], cc + line['line_dc']
            cr, cc = r_anchor - line['line_dr'], c_anchor - line['line_dc']
            while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                if board_pieces[cr][cc] == opponent: break
                occupation_updates[(cr, cc)] = player;
                cr, cc = cr - line['line_dr'], cc - line['line_dc']
    newly_captured_cells = []
    for (r, c), owner in occupation_updates.items():
        if board_territory[r][c] == EMPTY_SLOT: newly_captured_cells.append((r, c))
    for r_idx, c_idx in pieces_to_remove:
        board_pieces[r_idx][c_idx] = EMPTY_SLOT;
        center_x = c_idx * CELL_SIZE + PADDING + CELL_SIZE // 2;
        center_y = r_idx * CELL_SIZE + PADDING + CELL_SIZE // 2
        p_color = COLOR_PLAYER_B if player == PLAYER_BLACK else COLOR_PLAYER_W;
        create_particle_burst(center_x, center_y, p_color, count=50)
    for (r, c), p_color in occupation_updates.items(): board_territory[r][c] = p_color
    p_color_for_spawner = COLOR_PLAYER_B if player == PLAYER_BLACK else COLOR_PLAYER_W
    for r, c in newly_captured_cells: active_particle_spawners.append({'pos': (r, c), 'color': p_color_for_spawner})
    player_name = '青方' if player == PLAYER_BLACK else '品红方'
    message = f"{player_name}形成连线!"
    return has_line


def calculate_final_scores():
    scores = {PLAYER_BLACK: 0, PLAYER_WHITE: 0};
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            owner = board_territory[r][c]
            if owner == PLAYER_BLACK:
                scores[PLAYER_BLACK] += 1
            elif owner == PLAYER_WHITE:
                scores[PLAYER_WHITE] += 1
    return scores


def switch_player_and_set_message():
    global current_player, message
    if current_player == PLAYER_BLACK:
        current_player = PLAYER_WHITE
    else:
        current_player = PLAYER_BLACK
    if game_mode == GAME_MODE_AI and player_roles.get(current_player) == PLAYER_AI:
        message = "AI正在思考..."
    else:
        message = "轮到品红方" if current_player == PLAYER_WHITE else "轮到青方"


def screen_to_board_coords(x, y):
    if PADDING <= x < screen_width - PADDING and PADDING <= y < screen_height - INFO_AREA_HEIGHT - PADDING:
        return (y - PADDING) // CELL_SIZE, (x - PADDING) // CELL_SIZE
    return None, None


def is_valid_move(r, c, player_to_move):
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE): return False
    if board_pieces[r][c] != EMPTY_SLOT: return False
    opponent = PLAYER_WHITE if player_to_move == PLAYER_BLACK else PLAYER_BLACK
    if board_territory[r][c] == opponent: return False
    return True


def get_ai_move_task():
    global ai_move_result
    ai_game = AIGame();
    ai_game.current_move_number = current_move_number
    ai_game.current_player = ai_game.PLAYER_WHITE if current_player == PLAYER_WHITE else ai_game.PLAYER_BLACK
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p_pygame = board_pieces[r][c];
            t_pygame = board_territory[r][c]
            if p_pygame == PLAYER_BLACK:
                ai_game.board_pieces[r, c] = ai_game.PLAYER_BLACK
            elif p_pygame == PLAYER_WHITE:
                ai_game.board_pieces[r, c] = ai_game.PLAYER_WHITE
            else:
                ai_game.board_pieces[r, c] = 0
            if t_pygame == PLAYER_BLACK:
                ai_game.board_territory[r, c] = ai_game.PLAYER_BLACK
            elif t_pygame == PLAYER_WHITE:
                ai_game.board_territory[r, c] = ai_game.PLAYER_WHITE
            else:
                ai_game.board_territory[r, c] = 0
    mcts = MCTS_for_play(ai_game, args, AI_MODEL)
    action_probs = mcts.search(ai_game, add_noise=False)
    valid_moves = ai_game.get_valid_moves();
    action_probs = action_probs * valid_moves
    if np.sum(action_probs) == 0:
        valid_indices = np.where(valid_moves == 1)[0]
        ai_move_result = np.random.choice(valid_indices) if len(valid_indices) > 0 else None
    else:
        ai_move_result = np.argmax(action_probs)


# --- 绘制函数 (与您原始代码一致) ---
def draw_worn_line(surface, color, start, end, density=0.8):
    x1, y1 = start;
    x2, y2 = end
    if x1 == x2:
        for y in range(y1, y2):
            if random.random() < density: pygame.draw.rect(surface, color, (x1, y, 2, 2))
    elif y1 == y2:
        for x in range(x1, x2):
            if random.random() < density: pygame.draw.rect(surface, color, (x, y1, 2, 2))


def draw_cell_groove_border(surface, rect, highlight_color, shadow_color):
    thickness = 3
    for i in range(thickness):
        draw_worn_line(surface, highlight_color, (rect.left, rect.top + i), (rect.right, rect.top + i));
        draw_worn_line(surface, highlight_color, (rect.left + i, rect.top), (rect.left + i, rect.bottom))
    for i in range(thickness):
        draw_worn_line(surface, shadow_color, (rect.left, rect.bottom - 1 - i), (rect.right, rect.bottom - 1 - i));
        draw_worn_line(surface, shadow_color, (rect.right - 1 - i, rect.top), (rect.right - 1 - i, rect.bottom))


def draw_question_mark_glyph(surface, color, rect):
    center_x, center_y = rect.center;
    radius = rect.width * 0.2
    arc_rect = pygame.Rect(center_x - radius, center_y - radius * 1.5, radius * 2, radius * 2);
    pygame.draw.arc(surface, color, arc_rect, 0, math.pi * 1.2, 2)
    pygame.draw.line(surface, color, (center_x, center_y - radius * 0.5), (center_x, center_y + radius * 0.5), 2);
    pygame.draw.circle(surface, color, (center_x, center_y + radius * 1.5), 2)


def draw_inscription_circle(surface, center, radius, color):
    num_arcs = 4;
    arc_angle = (2 * math.pi) / num_arcs;
    gap_angle = math.radians(25);
    draw_angle = arc_angle - gap_angle
    for i in range(num_arcs):
        start_angle = i * arc_angle + gap_angle / 2;
        end_angle = start_angle + draw_angle
        pygame.draw.arc(surface, color, (center[0] - radius, center[1] - radius, radius * 2, radius * 2), -end_angle,
                        -start_angle, 2)


def draw_player_emblem(surface, center, player, emblem_color, size_multiplier=1.0):
    inner_radius = int(CELL_SIZE * 0.25 * size_multiplier);
    center_x, center_y = center
    if player == PLAYER_BLACK:
        arc_rect1 = pygame.Rect(center_x - inner_radius, center_y - inner_radius / 2, inner_radius * 2, inner_radius);
        arc_rect2 = pygame.Rect(center_x - inner_radius, center_y + inner_radius / 2, inner_radius * 2, inner_radius)
        pygame.draw.arc(surface, emblem_color, arc_rect1.move(0, -3 * size_multiplier), math.pi, 2 * math.pi, 2);
        pygame.draw.arc(surface, emblem_color, arc_rect2.move(0, 3 * size_multiplier), 0, math.pi, 2)
    else:
        for i in range(3):
            angle = 2 * math.pi / 3 * i - math.pi / 2;
            end_point_x = center_x + inner_radius * 0.8 * math.cos(angle);
            end_point_y = center_y + inner_radius * 0.8 * math.sin(angle)
            pygame.draw.line(surface, emblem_color, center, (end_point_x, end_point_y), 2)


def draw_2d_piece(surface, r, c, player, time):
    center_x = c * CELL_SIZE + PADDING + CELL_SIZE // 2;
    center_y = r * CELL_SIZE + PADDING + CELL_SIZE // 2
    color = COLOR_PLAYER_B if player == PLAYER_BLACK else COLOR_PLAYER_W
    pulse = (math.sin(time * 0.008 + r + c) + 1) / 2;
    outer_radius = int(CELL_SIZE * 0.35 + pulse * 4);
    inner_radius = int(CELL_SIZE * 0.25)
    pygame.draw.circle(surface, color, (center_x, center_y), outer_radius, width=int(3 + pulse * 2));
    pygame.draw.circle(surface, color, (center_x, center_y), inner_radius)
    draw_player_emblem(surface, (center_x, center_y), player, COLOR_BACKGROUND)


def draw_board_and_pieces(hover_coords):
    screen.fill(COLOR_BACKGROUND);
    time = pygame.time.get_ticks();
    global_pulse = (math.sin(time * 0.004) + 1) / 2
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rect = pygame.Rect(c * CELL_SIZE + PADDING, r * CELL_SIZE + PADDING, CELL_SIZE, CELL_SIZE);
            pygame.draw.rect(screen, COLOR_BOARD_BASE, rect)
            territory_owner = board_territory[r][c]
            if territory_owner != EMPTY_SLOT:
                color_bg = COLOR_TERRITORY_B_BG if territory_owner == PLAYER_BLACK else COLOR_TERRITORY_W_BG
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA);
                s.fill(color_bg);
                screen.blit(s, rect.topleft)
                pulse_amount = global_pulse if territory_owner == PLAYER_BLACK else (1.0 - global_pulse)
                base_color = COLOR_PLAYER_B if territory_owner == PLAYER_BLACK else COLOR_PLAYER_W
                pulsing_color = base_color.lerp(COLOR_TERRITORY_HIGHLIGHT, pulse_amount * 0.8)
                draw_inscription_circle(screen, rect.center, int(CELL_SIZE * 0.4), pulsing_color);
                draw_player_emblem(screen, rect.center, territory_owner, pulsing_color, 0.6)
            else:
                draw_question_mark_glyph(screen, COLOR_BOARD_GLYPH, rect)
            pulse_grid = (math.sin(time * 0.001 + r * 0.5 + c * 0.5) + 1) / 2
            highlight_color = COLOR_GRID_PULSE_DARK.lerp(COLOR_GRID_PULSE_LIGHT, pulse_grid)
            draw_cell_groove_border(screen, rect, highlight_color, COLOR_GROOVE_SHADOW)
    for spawner in active_particle_spawners:
        if random.random() < 0.25:
            r_s, c_s = spawner['pos'];
            center_x = c_s * CELL_SIZE + PADDING + CELL_SIZE // 2;
            center_y = r_s * CELL_SIZE + PADDING + CELL_SIZE // 2
            create_continuous_particle(center_x, center_y, spawner['color'])
    if hover_coords and hover_coords[0] is not None and not game_over and player_roles.get(
            current_player) == PLAYER_HUMAN:
        r, c = hover_coords
        if is_valid_move(r, c, current_player):
            rect = pygame.Rect(c * CELL_SIZE + PADDING, r * CELL_SIZE + PADDING, CELL_SIZE, CELL_SIZE);
            pygame.draw.rect(screen, COLOR_HOVER_HIGHLIGHT, rect, 2, border_radius=5)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board_pieces[r][c] != EMPTY_SLOT: draw_2d_piece(screen, r, c, board_pieces[r][c], time)


def draw_info_panel(mouse_pos, current_buttons):
    panel_rect = pygame.Rect(0, screen_height - INFO_AREA_HEIGHT, screen_width, INFO_AREA_HEIGHT);
    pygame.draw.rect(screen, COLOR_UI_BG, panel_rect);
    pygame.draw.rect(screen, COLOR_UI_BORDER, panel_rect, 4)
    padding_y = 10;
    padding_x = PADDING;
    line_height = game_font.get_height();
    line1_y = screen_height - INFO_AREA_HEIGHT + padding_y
    player_display_text = '青方' if current_player == PLAYER_BLACK else '品红方'
    player_role_text = ""
    if game_mode == GAME_MODE_AI:
        role = player_roles.get(current_player)
        if role == PLAYER_HUMAN:
            player_role_text = " (您)"
        elif role == PLAYER_AI:
            player_role_text = " (AI)"
    player_text_val = f"当前: {player_display_text}{player_role_text}"
    if ai_is_thinking:
        player_before_ai = PLAYER_BLACK if current_player == PLAYER_WHITE else PLAYER_WHITE
        player_display_text_before = '青方' if player_before_ai == PLAYER_BLACK else '品红方'
        player_text_val = f"当前: {player_display_text_before} (您)"
    text_surface_player = game_font.render(player_text_val, True, COLOR_UI_HIGHLIGHT);
    screen.blit(text_surface_player, (padding_x, line1_y))
    move_text_val = f"回合: {current_move_number}/{max_total_moves}";
    text_surface_move = game_font.render(move_text_val, True, COLOR_UI_TEXT)
    move_rect = text_surface_move.get_rect(right=screen_width - padding_x, top=line1_y);
    screen.blit(text_surface_move, move_rect)
    line2_y = line1_y + line_height + 5;
    scores = calculate_final_scores()
    score_text_val = f"青占: {scores.get(PLAYER_BLACK, 0)}  品红占: {scores.get(PLAYER_WHITE, 0)}"
    text_surface_score = game_font.render(score_text_val, True, COLOR_UI_TEXT);
    score_rect = text_surface_score.get_rect(centerx=screen_width // 2, top=line2_y);
    screen.blit(text_surface_score, score_rect)
    line3_y = line2_y + line_height + 5
    if message:
        msg_color = COLOR_UI_TEXT
        if "错误" in message or "不能" in message:
            msg_color = COLOR_UI_ERROR
        elif any(k in message for k in ["胜", "形成连线", "轮到", "先行", "思考中"]):
            msg_color = COLOR_UI_HIGHLIGHT
        text_render = game_font.render(message, True, msg_color);
        msg_rect = text_render.get_rect(centerx=screen_width // 2, top=line3_y);
        screen.blit(text_render, msg_rect)
    for button in current_buttons: button.draw(screen, mouse_pos)


# --- 主循环与菜单 ---
def main_game_loop(mode, hover_coords=None):
    global game_mode, player_roles, message, current_player, game_over, current_move_number, history, ai_is_thinking, ai_move_result
    game_mode = mode
    if game_mode == GAME_MODE_AI:
        player_roles = {PLAYER_BLACK: PLAYER_HUMAN, PLAYER_WHITE: PLAYER_AI}
    else:
        player_roles = {PLAYER_BLACK: PLAYER_HUMAN, PLAYER_WHITE: PLAYER_HUMAN}

    button_y = screen_height - 70;
    button_width = 120;
    button_height = 40
    total_button_width = button_width * 3 + 20 * 2;
    start_x = (screen_width - total_button_width) / 2
    game_buttons = [
        Button((start_x, button_y, button_width, button_height), "撤回一步", "undo", game_button_font),
        Button((start_x + button_width + 20, button_y, button_width, button_height), "重新开始", "restart",
               game_button_font),
        Button((start_x + 2 * (button_width + 20), button_y, button_width, button_height), "返回菜单", "main_menu",
               game_button_font)
    ]

    initialize_game_state(num_rounds_per_player_default)
    clock = pygame.time.Clock()
    ai_thread = None

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # AI Turn Logic
        if not game_over and player_roles.get(current_player) == PLAYER_AI and not ai_is_thinking:
            ai_is_thinking = True;
            message = "AI正在思考..."
            ai_thread = threading.Thread(target=get_ai_move_task);
            ai_thread.start()

        if ai_is_thinking and (ai_thread is None or not ai_thread.is_alive()):
            ai_is_thinking = False;
            action = ai_move_result
            if action is not None:
                r, c = action // BOARD_SIZE, action % BOARD_SIZE
                if is_valid_move(r, c, current_player):
                    place_piece(r, c, current_player);
                    lines_made = check_and_process_lines(current_player)
                    current_move_number += 1
                    switch_player_and_set_message()
            else:
                switch_player_and_set_message()
            ai_move_result = None
            history.append(get_state_snapshot())

        # Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

            action = None
            for button in game_buttons:
                if button.check_click(event): action = button.action; break
            if action == "restart": main_game_loop(mode); return
            if action == "main_menu": return
            if action == "undo":
                if not ai_is_thinking: undo_last_move()

            is_human_turn = not game_over and player_roles.get(current_player) == PLAYER_HUMAN
            if is_human_turn and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not any(
                    btn.rect.collidepoint(mouse_pos) for btn in game_buttons):
                clicked_row, clicked_col = screen_to_board_coords(*mouse_pos)
                if clicked_row is not None and is_valid_move(clicked_row, clicked_col, current_player):
                    place_piece(clicked_row, clicked_col, current_player)
                    lines_made = check_and_process_lines(current_player)
                    current_move_number += 1
                    switch_player_and_set_message()
                    history.append(get_state_snapshot())
                elif clicked_row is not None:
                    message = "错误: 无效落子"

        if not game_over and current_move_number >= max_total_moves:
            game_over = True;
            scores = calculate_final_scores()
            b_s, w_s = scores.get(PLAYER_BLACK, 0), scores.get(PLAYER_WHITE, 0)
            if b_s > w_s:
                message = f"游戏结束: 青方胜! (青:{b_s} 品红:{w_s})"
            elif w_s > b_s:
                message = f"游戏结束: 品红方胜! (品红:{w_s} 青:{b_s})"
            else:
                message = f"游戏结束: 平局! (均{b_s}分)"

        # Drawing
        draw_board_and_pieces(hover_coords)
        for p in particles[:]:
            p.update()
            if p.lifespan <= 0:
                particles.remove(p)
            else:
                p.draw(screen)
        draw_info_panel(mouse_pos, game_buttons)
        pygame.display.flip()
        clock.tick(60)


def main_menu_loop():
    btn_width, btn_height = 240, 60;
    btn_x = (screen_width - btn_width) / 2
    btn_y_single = screen_height / 2 - btn_height * 1.5 - 20;
    btn_y_dual = screen_height / 2;
    btn_y_exit = screen_height / 2 + btn_height * 1.5 + 20
    menu_buttons = [
        Button((btn_x, btn_y_single, btn_width, btn_height), "单人游戏 (VS AI)", GAME_MODE_AI, menu_button_font),
        Button((btn_x, btn_y_dual, btn_width, btn_height), "双人游戏", GAME_MODE_DUAL, menu_button_font),
        Button((btn_x, btn_y_exit, btn_width, btn_height), "退出游戏", "exit", menu_button_font),
    ]
    while True:
        mouse_pos = pygame.mouse.get_pos();
        screen.fill(COLOR_BACKGROUND)
        title_surf = title_font.render("扩展连线棋", True, COLOR_UI_HIGHLIGHT);
        title_rect = title_surf.get_rect(center=(screen_width / 2, screen_height / 4));
        screen.blit(title_surf, title_rect)
        for button in menu_buttons: button.draw(screen, mouse_pos)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit();sys.exit()
            for button in menu_buttons:
                action = button.check_click(event)
                if action:
                    if action == "exit": pygame.quit();sys.exit()
                    if action == GAME_MODE_AI and AI_MODEL is None: print("AI模型尚未加载或加载失败!");continue
                    main_game_loop(action)
        pygame.display.flip()


if __name__ == '__main__':
    load_ai_model()
    main_menu_loop()