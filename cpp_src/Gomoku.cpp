// file: cpp_src/Gomoku.cpp
#include "Gomoku.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <array>
#include <mutex>        // <-- 新增
//extern std::mutex g_io_mutex; // <-- 新增

// 构造函数
Gomoku::Gomoku(int board_size, int num_rounds)
    : board_size_(board_size),
      max_total_moves_(num_rounds * 2) {
    reset();
}

// 重置游戏状态
void Gomoku::reset() {
    board_pieces_.assign(board_size_, std::vector<int>(board_size_, EMPTY_SLOT));
    board_territory_.assign(board_size_, std::vector<int>(board_size_, EMPTY_SLOT));
    current_player_ = PLAYER_BLACK;
    current_move_number_ = 0;
}

// file: cpp_src/Gomoku.cpp
// ===================== 这是最关键的、必须应用的修改 =====================

// 拷贝构造函数 (实现真正的深拷贝)
Gomoku::Gomoku(const Gomoku& other)
    : board_size_(other.board_size_),
      max_total_moves_(other.max_total_moves_),
      current_player_(other.current_player_),
      current_move_number_(other.current_move_number_) {

    // 手动为二维向量分配空间并逐元素拷贝
    board_pieces_.assign(other.board_pieces_.begin(), other.board_pieces_.end());
    board_territory_.assign(other.board_territory_.begin(), other.board_territory_.end());
}

// 拷贝赋值运算符 (实现真正的深拷贝)
Gomoku& Gomoku::operator=(const Gomoku& other) {
    if (this == &other) {
        return *this;
    }

    // const 成员变量不能被赋值
    // 我们假设 board_size_ 和 max_total_moves_ 总是相等的

    current_player_ = other.current_player_;
    current_move_number_ = other.current_move_number_;

    // 手动为二维向量分配空间并逐元素拷贝
    board_pieces_.assign(other.board_pieces_.begin(), other.board_pieces_.end());
    board_territory_.assign(other.board_territory_.begin(), other.board_territory_.end());

    return *this;
}
// ===================== 修改结束 =====================

// 获取神经网络所需的状态张量 (扁平化)
std::vector<float> Gomoku::get_state() const {
    const int plane_size = board_size_ * board_size_;
    std::vector<float> state(6 * plane_size, 0.0f);

    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            int idx = r * board_size_ + c;
            state[0 * plane_size + idx] = (board_pieces_[r][c] == current_player_);
            state[1 * plane_size + idx] = (board_pieces_[r][c] == -current_player_);
            state[2 * plane_size + idx] = (board_territory_[r][c] == current_player_);
            state[3 * plane_size + idx] = (board_territory_[r][c] == -current_player_);
        }
    }

    float player_indicator = (current_player_ == PLAYER_BLACK) ? 1.0f : 0.0f;
    std::fill(state.begin() + 4 * plane_size, state.begin() + 5 * plane_size, player_indicator);

    float progress = (max_total_moves_ > 0) ? static_cast<float>(current_move_number_) / max_total_moves_ : 0.0f;
    std::fill(state.begin() + 5 * plane_size, state.begin() + 6 * plane_size, progress);

    return state;
}

// file: cpp_src/Gomoku.cpp (修正后)

std::vector<bool> Gomoku::get_valid_moves() const {
    std::vector<bool> valid_moves(board_size_ * board_size_);
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            // 一个合法的走法，必须同时满足“这个点是空的”和“这个点不是对方的领地”
            valid_moves[r * board_size_ + c] = (board_pieces_[r][c] == EMPTY_SLOT) && (board_territory_[r][c] != -current_player_);
        }
    }
    return valid_moves;
}

// 执行走子
void Gomoku::execute_move(int action) {
    int r = action / board_size_;
    int c = action % board_size_;
    // --- vvv 新增诊断日志 vvv ---
        /*{
            std::lock_guard<std::mutex> lock(g_io_mutex); // g_io_mutex 需要被包含进来
            std::cout << "[Gomoku::execute_move] Received action=" << action
                      << ". Checking square (" << r << "," << c << ")."
                      << " Current piece on that square: " << board_pieces_[r][c]
                      << std::endl;
        }*/
        // --- ^^^ 诊断日志结束 ^^^ ---

    // --- vvv 在这里新增一行诊断代码 vvv ---
        // std::cout << "[DEBUG] execute_move called with action=" << action << ". Target square (" << r << "," << c << ") has piece=" << board_pieces_[r][c] << std::endl;
        // --- ^^^ 新增代码结束 ^^^ ---

    if (r < 0 || r >= board_size_ || c < 0 || c >= board_size_ ||
        board_pieces_[r][c] != EMPTY_SLOT || board_territory_[r][c] == -current_player_) {
        throw std::invalid_argument("Invalid move action.");
    }

    board_pieces_[r][c] = current_player_;

    process_lines_and_territory(r, c);

    current_move_number_++;
    current_player_ *= -1; // 严格交替走子
}

// 检查游戏是否结束
std::pair<double, bool> Gomoku::get_game_ended() const {
    if (current_move_number_ >= max_total_moves_) {
        auto scores = calculate_scores();
        int black_score = scores.at(PLAYER_BLACK);
        int white_score = scores.at(PLAYER_WHITE);

        double winner_val = 0.0;
        if (black_score > white_score) winner_val = 1.0;
        else if (white_score > black_score) winner_val = -1.0;

        if (winner_val == 0.0) {
            return {0.001, true}; // 平局
        }
        return {winner_val, true};
    }
    return {0.0, false}; // 游戏未结束
}

// 核心逻辑：基于12个固定组合检查并处理三连珠和领地
void Gomoku::process_lines_and_territory(int r, int c) {
    const int player = current_player_;
    const int opponent = -player;

    // 定义相对于(r, c)的12个三连珠组合
    const std::array<std::array<std::pair<int, int>, 3>, 12> combinations = {{
        // Horizontal
        {{ {0, -2}, {0, -1}, {0, 0} }}, {{ {0, -1}, {0, 0}, {0, 1} }}, {{ {0, 0}, {0, 1}, {0, 2} }},
        // Vertical
        {{ {-2, 0}, {-1, 0}, {0, 0} }}, {{ {-1, 0}, {0, 0}, {1, 0} }}, {{ {0, 0}, {1, 0}, {2, 0} }},
        // Main Diagonal (\)
        {{ {-2, -2}, {-1, -1}, {0, 0} }}, {{ {-1, -1}, {0, 0}, {1, 1} }}, {{ {0, 0}, {1, 1}, {2, 2} }},
        // Anti-Diagonal (/)
        {{ {2, -2}, {1, -1}, {0, 0} }}, {{ {1, -1}, {0, 0}, {-1, 1} }}, {{ {0, 0}, {-1, 1}, {-2, 2} }}
    }};

    std::set<std::pair<int, int>> pieces_to_remove;
    bool axis_found[4] = {false, false, false, false}; // 0:H, 1:V, 2:MainDiag, 3:AntiDiag

    for (int i = 0; i < 12; ++i) {
        const auto& combo = combinations[i];
        std::pair<int, int> p1 = {r + combo[0].first, c + combo[0].second};
        std::pair<int, int> p2 = {r + combo[1].first, c + combo[1].second};
        std::pair<int, int> p3 = {r + combo[2].first, c + combo[2].second};

        // 检查边界
        if (p1.first < 0 || p1.first >= board_size_ || p1.second < 0 || p1.second >= board_size_ ||
            p2.first < 0 || p2.first >= board_size_ || p2.second < 0 || p2.second >= board_size_ ||
            p3.first < 0 || p3.first >= board_size_ || p3.second < 0 || p3.second >= board_size_) {
            continue;
        }

        // 检查是否形成三连珠
        if (board_pieces_[p1.first][p1.second] == player &&
            board_pieces_[p2.first][p2.second] == player &&
            board_pieces_[p3.first][p3.second] == player) {

            pieces_to_remove.insert(p1);
            pieces_to_remove.insert(p2);
            pieces_to_remove.insert(p3);
            axis_found[i / 3] = true; // 每3个组合属于一个轴线
        }
    }

    // 如果没有任何三连珠，直接返回
    if (pieces_to_remove.empty()) {
        return;
    }

    // 1. 消除棋子
    for (const auto& p : pieces_to_remove) {
        board_pieces_[p.first][p.second] = EMPTY_SLOT;
    }

    // 2. 占领领地
    const std::array<std::pair<int, int>, 4> directions = {{{0, 1}, {1, 0}, {1, 1}, {1, -1}}}; // H, V, MainDiag, AntiDiag
    for (int i = 0; i < 4; ++i) {
        if (axis_found[i]) {
            int dr = directions[i].first;
            int dc = directions[i].second;
            // 沿两个方向延伸
            for (int sign : {1, -1}) {
                int cr = r, cc = c;
                while (cr >= 0 && cr < board_size_ && cc >= 0 && cc < board_size_) {
                    if (board_pieces_[cr][cc] == opponent) break;
                    board_territory_[cr][cc] = player;
                    cr += sign * dr;
                    cc += sign * dc;
                }
            }
        }
    }
}

// 计算最终分数
std::map<int, int> Gomoku::calculate_scores() const {
    std::map<int, int> scores;
    scores[PLAYER_BLACK] = 0;
    scores[PLAYER_WHITE] = 0;
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            if (board_territory_[r][c] == PLAYER_BLACK) {
                scores[PLAYER_BLACK]++;
            } else if (board_territory_[r][c] == PLAYER_WHITE) {
                scores[PLAYER_WHITE]++;
            }
        }
    }
    return scores;
}

// --- 辅助函数实现 ---

int Gomoku::get_current_player() const {
    return current_player_;
}

int Gomoku::get_board_size() const {
    return board_size_;
}

void Gomoku::print_board() const {
    std::cout << "--- Board Pieces (Player: " << (current_player_ == 1 ? "B" : "W")
              << ", Move: " << current_move_number_ << ") ---" << std::endl;
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            char piece = '.';
            if (board_pieces_[r][c] == PLAYER_BLACK) piece = 'X';
            else if (board_pieces_[r][c] == PLAYER_WHITE) piece = 'O';
            std::cout << piece << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--- Board Territory ---" << std::endl;
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            char piece = '.';
            if (board_territory_[r][c] == PLAYER_BLACK) piece = 'B';
            else if (board_territory_[r][c] == PLAYER_WHITE) piece = 'W';
            std::cout << piece << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-------------------------" << std::endl;
}

// ... 在 print_board() 函数下方

// --- 新增辅助函数实现 ---
int Gomoku::get_move_number() const {
    return current_move_number_;
}