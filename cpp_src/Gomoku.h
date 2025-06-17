// file: cpp_src/Gomoku.h (最终版)
#pragma once

#include <vector>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <map>
#include <deque>
#include <array>

// 用于存储单步历史的位棋盘状态结构体
struct BitboardState {
    uint64_t black_stones[2];
    uint64_t white_stones[2];
    uint64_t black_territory[2];
    uint64_t white_territory[2];
};

class Gomoku {
public:
    // 静态常量
    static constexpr int EMPTY_SLOT = 0;
    static constexpr int PLAYER_BLACK = 1;
    static constexpr int PLAYER_WHITE = -1;

    // 默认构造函数
    Gomoku(int board_size = 9, int num_rounds = 25, int history_steps = 0);

    // 用于从特定状态恢复的构造函数
    Gomoku(
        int board_size,
        int max_total_moves,
        int current_player,
        int current_move_number,
        const uint64_t black_s[2],
        const uint64_t white_s[2],
        const uint64_t black_t[2],
        const uint64_t white_t[2],
        int history_steps = 0
    );

    // 深拷贝
    Gomoku(const Gomoku& other);
    Gomoku& operator=(const Gomoku& other);

    // 公有方法
    void reset();
    void execute_move(int action);
    std::vector<bool> get_valid_moves() const;
    std::pair<double, bool> get_game_ended() const;
    std::vector<float> get_state() const;
    int get_current_player() const;
    int get_board_size() const;
    int get_move_number() const;
    int get_territory_score() const;

    // 调试与辅助函数
    void print_board() const;
    bool is_on_board(int r, int c) const;
    bool is_occupied(int r, int c) const;

private:
    // 私有方法
    void process_patterns_and_territory(int r, int c);
    std::map<int, int> calculate_scores() const;

    // 成员变量
    const int board_size_;
    const int max_total_moves_;
    const int history_steps_;
    int current_player_;
    int current_move_number_;
    int last_move_action_;

    // 位棋盘核心数据
    uint64_t black_stones_[2];
    uint64_t white_stones_[2];
    uint64_t black_territory_[2];
    uint64_t white_territory_[2];

    // 历史状态队列
    std::deque<BitboardState> history_;
};