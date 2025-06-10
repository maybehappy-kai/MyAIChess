import pygame
import sys
#from game import Game

# --- 常量和绘制函数 (无需改动) ---
BOARD_SIZE = 9;
CELL_SIZE = 60;
PADDING = 30;
INFO_PANEL_HEIGHT = 100;
SCREEN_WIDTH = BOARD_SIZE * CELL_SIZE + 2 * PADDING;
SCREEN_HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * PADDING + INFO_PANEL_HEIGHT;
COLOR_BACKGROUND = (20, 30, 40);
COLOR_BOARD = (40, 50, 60);
COLOR_GRID = (80, 90, 100);
COLOR_BLACK_PIECE = (10, 10, 10);
COLOR_WHITE_PIECE = (240, 240, 240);
COLOR_BLACK_TERRITORY = (0, 150, 150, 100);
COLOR_WHITE_TERRITORY = (200, 0, 200, 100);
COLOR_HOVER_HIGHLIGHT = (255, 255, 0);
COLOR_TEXT = (220, 220, 220);
COLOR_INFO_BG = (30, 40, 50)


def draw_board(screen, game):
    board_rect = pygame.Rect(PADDING, PADDING, BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE);
    pygame.draw.rect(screen, COLOR_BOARD, board_rect)
    for r in range(game.BOARD_SIZE):
        for c in range(game.BOARD_SIZE):
            cell_rect = pygame.Rect(PADDING + c * CELL_SIZE, PADDING + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            territory_owner = game.board_territory[r, c]
            if territory_owner != game.EMPTY_SLOT:
                color = COLOR_BLACK_TERRITORY if territory_owner == game.PLAYER_BLACK else COLOR_WHITE_TERRITORY
                s = pygame.Surface(cell_rect.size, pygame.SRCALPHA);
                s.fill(color);
                screen.blit(s, cell_rect.topleft)
            pygame.draw.rect(screen, COLOR_GRID, cell_rect, 1)
            piece_owner = game.board_pieces[r, c]
            if piece_owner != game.EMPTY_SLOT:
                color = COLOR_BLACK_PIECE if piece_owner == game.PLAYER_BLACK else COLOR_WHITE_PIECE
                pygame.draw.circle(screen, color, cell_rect.center, int(CELL_SIZE * 0.4))


def draw_info_panel(screen, game, font):
    panel_rect = pygame.Rect(0, SCREEN_HEIGHT - INFO_PANEL_HEIGHT, SCREEN_WIDTH, INFO_PANEL_HEIGHT);
    pygame.draw.rect(screen, COLOR_INFO_BG, panel_rect)
    if not game.game_over:
        player_char = 'X (黑方)' if game.current_player == game.PLAYER_BLACK else 'O (白方)';
        text = f"当前玩家: {player_char}"
    else:
        text = "游戏结束"
    text_surf = font.render(text, True, COLOR_TEXT);
    screen.blit(text_surf, (PADDING, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 10))
    scores = game._calculate_scores();
    score_text = f"分数 - 黑: {scores[game.PLAYER_BLACK]} | 白: {scores[game.PLAYER_WHITE]}";
    score_surf = font.render(score_text, True, COLOR_TEXT);
    screen.blit(score_surf, (PADDING, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 40))
    if game.game_over:
        winner = "黑方 'X' 获胜!" if game.winner == game.PLAYER_BLACK else "白方 'O' 获胜!" if game.winner == game.PLAYER_WHITE else "平局!";
        winner_text = f"结果: {winner} (按 R 键重新开始)";
        winner_surf = font.render(winner_text, True, COLOR_HOVER_HIGHLIGHT);
        screen.blit(winner_surf, (PADDING, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 70))
    else:
        restart_surf = font.render("按 R 键可随时重新开始", True, COLOR_GRID); screen.blit(restart_surf, (
        SCREEN_WIDTH - PADDING - restart_surf.get_width(), SCREEN_HEIGHT - 30))


def screen_to_board_coords(pos):
    x, y = pos
    if PADDING <= x < SCREEN_WIDTH - PADDING and PADDING <= y < SCREEN_HEIGHT - INFO_PANEL_HEIGHT - PADDING: return (
                                                                                                                                y - PADDING) // CELL_SIZE, (
                                                                                                                                x - PADDING) // CELL_SIZE
    return None, None


def main():
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("可交互逻辑棋盘 (最终解决版)")
    font = pygame.font.SysFont("Microsoft YaHei", 20, bold=True)
    clock = pygame.time.Clock()
    game = Game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --- 最终修正逻辑 ---

            # 方案A：监听物理按键（适用于英文输入法）
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()

            # 方案B：监听文本编辑事件（适用于中文输入法）
            if event.type == pygame.TEXTEDITING:
                if event.text.lower() == 'r':
                    game.reset()

            # 处理鼠标点击
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not game.game_over:
                    r, c = screen_to_board_coords(event.pos)
                    if r is not None:
                        action = r * game.BOARD_SIZE + c
                        if game.get_valid_moves()[action]:
                            game.execute_move(action)

        # --- 绘制 Section ---
        screen.fill(COLOR_BACKGROUND)
        draw_board(screen, game)
        draw_info_panel(screen, game, font)

        if not game.game_over:
            hover_r, hover_c = screen_to_board_coords(pygame.mouse.get_pos())
            if hover_r is not None:
                action_idx = hover_r * game.BOARD_SIZE + hover_c
                if game.get_valid_moves()[action_idx]:
                    pygame.draw.rect(screen, COLOR_HOVER_HIGHLIGHT, (
                    PADDING + hover_c * CELL_SIZE, PADDING + hover_r * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()