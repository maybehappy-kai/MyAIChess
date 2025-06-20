// file: cpp_src/Gomoku.cpp (绝对最终版)

#include "Gomoku.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <map>
#include <array>

// 为了兼容性，定义popcount
#ifdef _MSC_VER
#include <intrin.h>
#define popcount __popcnt64
#else
#define popcount __builtin_popcountll
#endif

// ===================================================================
// 构造函数与拷贝/赋值函数
// ===================================================================

Gomoku::Gomoku(int board_size, int num_rounds, int history_steps)
    : board_size_(board_size),
      max_total_moves_(num_rounds * 2),
      history_steps_(history_steps)
{
    if (board_size * board_size > 128)
    {
        throw std::invalid_argument("Board size is too large for the bitboard implementation (max 11x11).");
    }
    reset();
}

Gomoku::Gomoku(
    int board_size, int max_total_moves, int current_player, int current_move_number,
    const uint64_t black_s[2], const uint64_t white_s[2],
    const uint64_t black_t[2], const uint64_t white_t[2],
    int history_steps)
    : board_size_(board_size),
      max_total_moves_(max_total_moves),
      history_steps_(history_steps),
      current_player_(current_player),
      current_move_number_(current_move_number),
      last_move_action_(-1)
{
    for (int i = 0; i < 2; ++i)
    {
        this->black_stones_[i] = black_s[i];
        this->white_stones_[i] = white_s[i];
        this->black_territory_[i] = black_t[i];
        this->white_territory_[i] = white_t[i];
    }
    // history_.clear();
    BitboardState current_state;
    for (int i = 0; i < 2; ++i)
    {
        current_state.black_stones[i] = this->black_stones_[i];
        current_state.white_stones[i] = this->white_stones_[i];
        current_state.black_territory[i] = this->black_territory_[i];
        current_state.white_territory[i] = this->white_territory_[i];
    }
    /*for (int i = 0; i < history_steps_ + 1; ++i) {
        history_.push_back(current_state);
    }*/
}

Gomoku::Gomoku(const Gomoku &other)
    : board_size_(other.board_size_),
      max_total_moves_(other.max_total_moves_),
      history_steps_(other.history_steps_),
      current_player_(other.current_player_),
      current_move_number_(other.current_move_number_),
      last_move_action_(other.last_move_action_)
{
    for (int i = 0; i < 2; ++i)
    {
        this->black_stones_[i] = other.black_stones_[i];
        this->white_stones_[i] = other.white_stones_[i];
        this->black_territory_[i] = other.black_territory_[i];
        this->white_territory_[i] = other.white_territory_[i];
    }
    // this->history_ = other.history_;
}

Gomoku &Gomoku::operator=(const Gomoku &other)
{
    if (this != &other)
    {
        // const 成员变量不能被赋值，所以它们保持不变
        current_player_ = other.current_player_;
        current_move_number_ = other.current_move_number_;
        last_move_action_ = other.last_move_action_;
        for (int i = 0; i < 2; ++i)
        {
            this->black_stones_[i] = other.black_stones_[i];
            this->white_stones_[i] = other.white_stones_[i];
            this->black_territory_[i] = other.black_territory_[i];
            this->white_territory_[i] = other.white_territory_[i];
        }
        // this->history_ = other.history_;
    }
    return *this;
}

// ===================================================================
// 核心游戏逻辑
// ===================================================================

void Gomoku::reset()
{
    black_stones_[0] = black_stones_[1] = 0;
    white_stones_[0] = white_stones_[1] = 0;
    black_territory_[0] = black_territory_[1] = 0;
    white_territory_[0] = white_territory_[1] = 0;
    current_player_ = PLAYER_BLACK;
    current_move_number_ = 0;
    last_move_action_ = -1;
    // history_.clear();
    BitboardState empty_state = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    /*for (int i = 0; i < history_steps_ + 1; ++i) {
        history_.push_front(empty_state);
    }*/
}

void Gomoku::execute_move(int action)
{
    const int r = action / board_size_;
    const int c = action % board_size_;
    if (!is_on_board(r, c) || is_occupied(r, c))
    {
        throw std::invalid_argument("Invalid move action.");
    }

    /*BitboardState current_state_for_history;
    for (int i = 0; i < 2; ++i) {
        current_state_for_history.black_stones[i] = this->black_stones_[i];
        current_state_for_history.white_stones[i] = this->white_stones_[i];
        current_state_for_history.black_territory[i] = this->black_territory_[i];
        current_state_for_history.white_territory[i] = this->white_territory_[i];
    }
    history_.push_front(current_state_for_history);
    if (history_.size() > static_cast<size_t>(history_steps_ + 1)) {
        history_.pop_back();
    }*/

    const int pos = r * board_size_ + c;
    const int index = pos / 64;
    const uint64_t mask = 1ULL << (pos % 64);

    if (current_player_ == PLAYER_BLACK)
        black_stones_[index] |= mask;
    else
        white_stones_[index] |= mask;

    process_patterns_and_territory(r, c);

    current_move_number_++;
    current_player_ *= -1;
    last_move_action_ = action;
}

void Gomoku::process_patterns_and_territory(int r, int c)
{
    uint64_t *player_stones = (current_player_ == PLAYER_BLACK) ? black_stones_ : white_stones_;
    uint64_t *opponent_stones = (current_player_ == PLAYER_BLACK) ? white_stones_ : black_stones_;
    uint64_t *player_territory = (current_player_ == PLAYER_BLACK) ? black_territory_ : white_territory_;
    uint64_t *opponent_territory = (current_player_ == PLAYER_BLACK) ? white_territory_ : black_territory_;
    const std::array<std::array<std::pair<int, int>, 3>, 12> combinations = {{{{{0, -2}, {0, -1}, {0, 0}}}, {{{0, -1}, {0, 0}, {0, 1}}}, {{{0, 0}, {0, 1}, {0, 2}}}, {{{-2, 0}, {-1, 0}, {0, 0}}}, {{{-1, 0}, {0, 0}, {1, 0}}}, {{{0, 0}, {1, 0}, {2, 0}}}, {{{-2, -2}, {-1, -1}, {0, 0}}}, {{{-1, -1}, {0, 0}, {1, 1}}}, {{{0, 0}, {1, 1}, {2, 2}}}, {{{2, -2}, {1, -1}, {0, 0}}}, {{{1, -1}, {0, 0}, {-1, 1}}}, {{{0, 0}, {-1, 1}, {-2, 2}}}}};
    uint64_t to_remove[2] = {0, 0};
    bool axis_found[4] = {false, false, false, false};

    for (int i = 0; i < 12; ++i)
    {
        const auto &combo = combinations[i];
        bool all_found = true;
        uint64_t temp_masks[2] = {0, 0};
        for (const auto &p_offset : combo)
        {
            int pr = r + p_offset.first, pc = c + p_offset.second;
            if (!is_on_board(pr, pc))
            {
                all_found = false;
                break;
            }
            int pos = pr * board_size_ + pc, index = pos / 64;
            uint64_t mask = 1ULL << (pos % 64);
            if (!(player_stones[index] & mask))
            {
                all_found = false;
                break;
            }
            temp_masks[index] |= mask;
        }
        if (all_found)
        {
            to_remove[0] |= temp_masks[0];
            to_remove[1] |= temp_masks[1];
            axis_found[i / 3] = true;
        }
    }

    if (to_remove[0] == 0 && to_remove[1] == 0)
        return;
    player_stones[0] &= ~to_remove[0];
    player_stones[1] &= ~to_remove[1];

    const std::array<std::pair<int, int>, 4> directions = {{{0, 1}, {1, 0}, {1, 1}, {1, -1}}};
    for (int i = 0; i < 4; ++i)
    {
        if (axis_found[i])
        {
            int dr = directions[i].first, dc = directions[i].second;
            for (int sign : {1, -1})
            {
                int cr = r, cc = c;
                while (is_on_board(cr, cc))
                {
                    int pos = cr * board_size_ + cc, index = pos / 64;
                    uint64_t mask = 1ULL << (pos % 64);
                    if (opponent_stones[index] & mask)
                        break;
                    opponent_territory[index] &= ~mask;
                    player_territory[index] |= mask;
                    cr += sign * dr;
                    cc += sign * dc;
                }
            }
        }
    }
}

// ===================================================================
// 数据获取与辅助函数
// ===================================================================

std::vector<bool> Gomoku::get_valid_moves() const
{
    std::vector<bool> valid_moves(board_size_ * board_size_, false);
    uint64_t occupied[2] = {black_stones_[0] | white_stones_[0], black_stones_[1] | white_stones_[1]};
    const uint64_t *opponent_territory = (current_player_ == PLAYER_BLACK) ? white_territory_ : black_territory_;
    uint64_t valid_mask[2] = {~(occupied[0] | opponent_territory[0]), ~(occupied[1] | opponent_territory[1])};

    for (int i = 0; i < board_size_ * board_size_; ++i)
    {
        const int index = i / 64;
        const uint64_t mask = 1ULL << (i % 64);
        if (valid_mask[index] & mask)
        {
            valid_moves[i] = true;
        }
    }
    return valid_moves;
}

// file: cpp_src/Gomoku.cpp

// 请用这个新版本的函数，完整替换旧的同名函数
std::pair<double, bool> Gomoku::get_game_ended() const
{
    // 1. 检查总步数是否达到上限
    if (current_move_number_ >= max_total_moves_)
    {
        auto scores = calculate_scores();
        int black_score = scores.at(PLAYER_BLACK);
        int white_score = scores.at(PLAYER_WHITE);
        double winner_val = 0.0;
        if (black_score > white_score)
            winner_val = 1.0;
        else if (white_score > black_score)
            winner_val = -1.0;
        // 如果平局，返回一个极小值，避免被误认为是未结束
        return {winner_val == 0.0 ? 0.001 : winner_val, true};
    }

    // 2.【新增的核心逻辑】检查当前玩家是否还有合法的落子点
    auto valid_moves = get_valid_moves();
    bool has_valid_move = false;
    for (bool move_is_valid : valid_moves)
    {
        if (move_is_valid)
        {
            has_valid_move = true;
            break;
        }
    }

    if (!has_valid_move)
    {
        // 如果当前玩家无路可走，则对方获胜
        // 对方玩家的值是 -current_player_ (例如，当前是黑1，则白-1获胜)
        return {static_cast<double>(-current_player_), true};
    }

    // 3. 如果以上条件都不满足，说明游戏尚未结束
    return {0.0, false};
}

std::vector<float> Gomoku::get_state(const std::deque<BitboardState> &history) const
{
    const int plane_size = board_size_ * board_size_;
    const int total_channels = (history_steps_ + 1) * 4 + 4;
    std::vector<float> state(total_channels * plane_size, 0.0f);

    for (int t = 0; t <= history_steps_; ++t)
    {
        if (static_cast<size_t>(t) >= history.size())
            continue;

        // 注意：这里player_at_t的计算逻辑可能需要根据调用者如何构建history进行调整
        // 假设调用者提供的history[0]是当前状态的前一步(T-1)，history[1]是T-2...
        // 那么history[t]对应的玩家是 current_player_ * (-1)^(t+1)
        const int player_at_t = (t % 2 == 0) ? -current_player_ : current_player_;

        const BitboardState &historical_state = history[t];
        const uint64_t *p1_s = (player_at_t == PLAYER_BLACK) ? historical_state.black_stones : historical_state.white_stones;
        const uint64_t *p2_s = (player_at_t == PLAYER_BLACK) ? historical_state.white_stones : historical_state.black_stones;
        const uint64_t *p1_t = (player_at_t == PLAYER_BLACK) ? historical_state.black_territory : historical_state.white_territory;
        const uint64_t *p2_t = (player_at_t == PLAYER_BLACK) ? historical_state.white_territory : historical_state.black_territory;
        const int channel_offset = (t + 1) * 4; // T-1, T-2...从第4个通道开始
        for (int i = 0; i < plane_size; ++i)
        {
            const int index = i / 64;
            const uint64_t mask = 1ULL << (i % 64);
            if (p1_s[index] & mask)
                state[(channel_offset + 0) * plane_size + i] = 1.0f;
            if (p2_s[index] & mask)
                state[(channel_offset + 1) * plane_size + i] = 1.0f;
            if (p1_t[index] & mask)
                state[(channel_offset + 2) * plane_size + i] = 1.0f;
            if (p2_t[index] & mask)
                state[(channel_offset + 3) * plane_size + i] = 1.0f;
        }
    }

    // 填充当前状态 (T=0)
    const uint64_t *current_p1_s = (current_player_ == PLAYER_BLACK) ? black_stones_ : white_stones_;
    const uint64_t *current_p2_s = (current_player_ == PLAYER_BLACK) ? white_stones_ : black_stones_;
    const uint64_t *current_p1_t = (current_player_ == PLAYER_BLACK) ? black_territory_ : white_territory_;
    const uint64_t *current_p2_t = (current_player_ == PLAYER_BLACK) ? white_territory_ : black_territory_;
    for (int i = 0; i < plane_size; ++i)
    {
        const int index = i / 64;
        const uint64_t mask = 1ULL << (i % 64);
        if (current_p1_s[index] & mask)
            state[0 * plane_size + i] = 1.0f;
        if (current_p2_s[index] & mask)
            state[1 * plane_size + i] = 1.0f;
        if (current_p1_t[index] & mask)
            state[2 * plane_size + i] = 1.0f;
        if (current_p2_t[index] & mask)
            state[3 * plane_size + i] = 1.0f;
    }

    // 填充元数据通道
    const int meta_offset = (history_steps_ + 1) * 4;
    const float player_indicator = (current_player_ == PLAYER_BLACK) ? 1.0f : 0.0f;
    std::fill(state.begin() + (meta_offset + 0) * plane_size, state.begin() + (meta_offset + 1) * plane_size, player_indicator);
    const float progress = (max_total_moves_ > 0) ? static_cast<float>(current_move_number_) / max_total_moves_ : 0.0f;
    std::fill(state.begin() + (meta_offset + 1) * plane_size, state.begin() + (meta_offset + 2) * plane_size, progress);
    if (last_move_action_ != -1)
    {
        state[(meta_offset + 2) * plane_size + last_move_action_] = 1.0f;
    }

    // 填充“领地变化”通道
    if (!history.empty())
    {
        const BitboardState &prev_board = history.front(); // T-1步的状态
        const int last_player = -current_player_;
        const uint64_t *territory_curr = (last_player == PLAYER_BLACK) ? black_territory_ : white_territory_;
        const uint64_t *territory_prev = (last_player == PLAYER_BLACK) ? prev_board.black_territory : prev_board.white_territory;
        uint64_t changed[2] = {territory_curr[0] & ~territory_prev[0], territory_curr[1] & ~territory_prev[1]};
        for (int i = 0; i < plane_size; ++i)
        {
            const int index = i / 64;
            const uint64_t mask = 1ULL << (i % 64);
            if (changed[index] & mask)
            {
                state[(meta_offset + 3) * plane_size + i] = 1.0f;
            }
        }
    }
    return state;
}

/*std::vector<float> Gomoku::get_state() const {
    const int plane_size = board_size_ * board_size_;
    const int total_channels = (history_steps_ + 1) * 4 + 4;
    std::vector<float> state(total_channels * plane_size, 0.0f);

    for (int t = 0; t <= history_steps_; ++t) {
        if (static_cast<size_t>(t) >= history_.size()) continue;
        const int player_at_t = (t % 2 == 0) ? current_player_ : -current_player_;
        const BitboardState& historical_state = history_[t];
        const uint64_t* p1_s = (player_at_t == PLAYER_BLACK) ? historical_state.black_stones : historical_state.white_stones;
        const uint64_t* p2_s = (player_at_t == PLAYER_BLACK) ? historical_state.white_stones : historical_state.black_stones;
        const uint64_t* p1_t = (player_at_t == PLAYER_BLACK) ? historical_state.black_territory : historical_state.white_territory;
        const uint64_t* p2_t = (player_at_t == PLAYER_BLACK) ? historical_state.white_territory : historical_state.black_territory;
        const int channel_offset = t * 4;
        for (int i = 0; i < plane_size; ++i) {
            const int index = i / 64;
            const uint64_t mask = 1ULL << (i % 64);
            if (p1_s[index] & mask) state[(channel_offset + 0) * plane_size + i] = 1.0f;
            if (p2_s[index] & mask) state[(channel_offset + 1) * plane_size + i] = 1.0f;
            if (p1_t[index] & mask) state[(channel_offset + 2) * plane_size + i] = 1.0f;
            if (p2_t[index] & mask) state[(channel_offset + 3) * plane_size + i] = 1.0f;
        }
    }

    const int meta_offset = (history_steps_ + 1) * 4;
    const float player_indicator = (current_player_ == PLAYER_BLACK) ? 1.0f : 0.0f;
    std::fill(state.begin() + (meta_offset + 0) * plane_size, state.begin() + (meta_offset + 1) * plane_size, player_indicator);
    const float progress = (max_total_moves_ > 0) ? static_cast<float>(current_move_number_) / max_total_moves_ : 0.0f;
    std::fill(state.begin() + (meta_offset + 1) * plane_size, state.begin() + (meta_offset + 2) * plane_size, progress);
    if (last_move_action_ != -1) {
        state[(meta_offset + 2) * plane_size + last_move_action_] = 1.0f;
    }
    if (history_.size() > 1) {
        const int last_player = -current_player_;
        const BitboardState& current_board = history_[0];
        const BitboardState& prev_board = history_[1];
        const uint64_t* territory_curr = (last_player == PLAYER_BLACK) ? current_board.black_territory : current_board.white_territory;
        const uint64_t* territory_prev = (last_player == PLAYER_BLACK) ? prev_board.black_territory : prev_board.white_territory;
        uint64_t changed[2] = { territory_curr[0] & ~territory_prev[0], territory_curr[1] & ~territory_prev[1] };
        for (int i = 0; i < plane_size; ++i) {
            const int index = i / 64;
            const uint64_t mask = 1ULL << (i % 64);
            if (changed[index] & mask) {
                state[(meta_offset + 3) * plane_size + i] = 1.0f;
            }
        }
    }
    return state;
}*/

std::map<int, int> Gomoku::calculate_scores() const
{
    std::map<int, int> scores;
    scores[PLAYER_BLACK] = popcount(black_territory_[0]) + popcount(black_territory_[1]);
    scores[PLAYER_WHITE] = popcount(white_territory_[0]) + popcount(white_territory_[1]);
    return scores;
}

int Gomoku::get_territory_score() const
{
    int p1_score = (current_player_ == PLAYER_BLACK) ? (popcount(black_territory_[0]) + popcount(black_territory_[1])) : (popcount(white_territory_[0]) + popcount(white_territory_[1]));
    int p2_score = (current_player_ == PLAYER_BLACK) ? (popcount(white_territory_[0]) + popcount(white_territory_[1])) : (popcount(black_territory_[0]) + popcount(black_territory_[1]));
    return p1_score - p2_score;
}

int Gomoku::get_current_player() const { return current_player_; }
int Gomoku::get_board_size() const { return board_size_; }
int Gomoku::get_move_number() const { return current_move_number_; }

bool Gomoku::is_on_board(int r, int c) const
{
    return r >= 0 && r < board_size_ && c >= 0 && c < board_size_;
}

// 【最终修正】确保此函数被正确实现为Gomoku类的成员
bool Gomoku::is_occupied(int r, int c) const
{
    const int pos = r * board_size_ + c;
    const int index = pos / 64;
    const uint64_t mask = 1ULL << (pos % 64);
    return (black_stones_[index] & mask) || (white_stones_[index] & mask);
}

// 文件: cpp_src/Gomoku.cpp

BitboardState Gomoku::get_bitboard_state() const
{
    BitboardState current_state;
    for (int i = 0; i < 2; ++i)
    {
        current_state.black_stones[i] = this->black_stones_[i];
        current_state.white_stones[i] = this->white_stones_[i];
        current_state.black_territory[i] = this->black_territory_[i];
        current_state.white_territory[i] = this->white_territory_[i];
    }
    return current_state;
}

// file: cpp_src/Gomoku.cpp
// 用这个版本替换旧的 print_board()
void Gomoku::print_board() const
{
    std::cout << "--- Board (Next to move: " << (current_player_ == 1 ? "B 'X'" : "W 'O'")
              << ", Total Moves: " << current_move_number_ << ") ---\n";

    int black_score = popcount(black_territory_[0]) + popcount(black_territory_[1]);
    int white_score = popcount(white_territory_[0]) + popcount(white_territory_[1]);
    std::cout << "--- Territory: Black(x) = " << black_score
              << ", White(o) = " << white_score << " ---\n";

    // 打印列号
    std::cout << "  ";
    for (int c = 0; c < board_size_; ++c)
    {
        std::cout << c << " ";
    }
    std::cout << "\n";

    for (int r = 0; r < board_size_; ++r)
    {
        std::cout << r << " "; // 打印行号
        for (int c = 0; c < board_size_; ++c)
        {
            const int pos = r * board_size_ + c;
            const int index = pos / 64;
            const uint64_t mask = 1ULL << (pos % 64);
            char piece = '.';
            if (black_stones_[index] & mask)
                piece = 'X';
            else if (white_stones_[index] & mask)
                piece = 'O';
            else if (black_territory_[index] & mask)
                piece = 'x';
            else if (white_territory_[index] & mask)
                piece = 'o';
            std::cout << piece << " ";
        }
        std::cout << std::endl;
    }
}

const uint64_t *Gomoku::get_player_stones_bitboard() const
{
    return (current_player_ == PLAYER_BLACK) ? black_stones_ : white_stones_;
}

const uint64_t *Gomoku::get_player_territory_bitboard() const
{
    return (current_player_ == PLAYER_BLACK) ? black_territory_ : white_territory_;
}