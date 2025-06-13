// file: cpp_src/Gomoku.h
#pragma once

#include <vector>
#include <utility>   // for std::pair
#include <stdexcept> // for std::invalid_argument
#include <set>       // for std::set
#include <map>       // for std::map

class Gomoku {
public:
    // 静态常量，用于定义棋盘状态
    static constexpr int EMPTY_SLOT = 0;
    static constexpr int PLAYER_BLACK = 1;
    static constexpr int PLAYER_WHITE = -1;
    // 构造函数与重置
    Gomoku(int board_size = 9, int num_rounds = 25);
    void reset();

    // ====================== 在这里新增构造函数声明 ======================
        Gomoku(
            int board_size,
            int max_total_moves,
            int current_player,
            int current_move_number,
            const std::vector<std::vector<int>>& board_pieces,
            const std::vector<std::vector<int>>& board_territory
        );
        // ================================================================

    // 公有方法，复刻并优化后的游戏逻辑
    void execute_move(int action);
    std::vector<bool> get_valid_moves() const;
    std::pair<double, bool> get_game_ended() const;
    std::vector<float> get_state() const; // 返回一个扁平化的向量
    int get_current_player() const;

    // 方便调试和查看状态的辅助函数
    // ...
    int get_board_size() const;
    int get_move_number() const; // <-- 新增这一行
    void print_board() const;
    // ...

    // 实现深拷贝
    Gomoku(const Gomoku& other); // 拷贝构造函数
    Gomoku& operator=(const Gomoku& other); // 拷贝赋值运算符

private:
    // 私有方法，用于内部逻辑
    void process_lines_and_territory(int r, int c);
    std::map<int, int> calculate_scores() const;

    // 私有成员变量
    const int board_size_;
    const int max_total_moves_;

    int current_player_;
    int current_move_number_;

    std::vector<std::vector<int>> board_pieces_;
    std::vector<std::vector<int>> board_territory_;


};