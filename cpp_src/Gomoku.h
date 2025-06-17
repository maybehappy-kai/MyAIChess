// file: cpp_src/Gomoku.h
#pragma once

#include <vector>
#include <utility>
#include <stdexcept>
#include <set>
#include <map>
#include <deque> // <--- 引入 deque

// vvvvvv 新增历史状态结构体 vvvvvv
struct BoardState {
    std::vector<std::vector<int>> pieces;
    std::vector<std::vector<int>> territory;
};
// ^^^^^^ 新增历史状态结构体 ^^^^^^

class Gomoku {
public:
    // 静态常量
    static constexpr int EMPTY_SLOT = 0;
    static constexpr int PLAYER_BLACK = 1;
    static constexpr int PLAYER_WHITE = -1;

    // vvvvvv 修改构造函数声明 vvvvvv
    // 接收所有配置参数，并提供默认值
    Gomoku(int board_size = 9, int num_rounds = 25, int history_steps = 0);
    // ^^^^^^ 修改构造函数声明 ^^^^^^

    void reset();

    // 保留这个用于特定状态恢复的构造函数
    Gomoku(
        int board_size,
        int max_total_moves,
        int current_player,
        int current_move_number,
        const std::vector<std::vector<int>>& board_pieces,
        const std::vector<std::vector<int>>& board_territory,
        int history_steps = 0 // 添加这个参数，默认为0
    );

    // 公有方法
    void execute_move(int action);
    std::vector<bool> get_valid_moves() const;
    std::pair<double, bool> get_game_ended() const;
    std::vector<float> get_state() const;
    int get_current_player() const;
    int get_board_size() const;
    int get_move_number() const;
    int get_territory_score() const;
    void print_board() const;

    // 深拷贝
    Gomoku(const Gomoku& other);
    Gomoku& operator=(const Gomoku& other);

private:
    // 私有方法
    void process_lines_and_territory(int r, int c);
    std::map<int, int> calculate_scores() const;

    // vvvvvv 修改成员变量 vvvvvv
    // 私有成员变量
    const int board_size_;
    const int max_total_moves_;
    const int history_steps_; // 新增：存储历史步数

    int current_player_;
    int current_move_number_;

    std::vector<std::vector<int>> board_pieces_;
    std::vector<std::vector<int>> board_territory_;

    int last_move_action_;
    std::vector<std::vector<int>> previous_board_territory_;

    // 新增历史记录队列
    std::deque<BoardState> history_;
    // ^^^^^^ 修改成员变量 ^^^^^^
};