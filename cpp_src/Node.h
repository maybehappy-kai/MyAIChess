// file: cpp_src/Node.h
#pragma once

#include "Gomoku.h" // 节点需要知道游戏状态
#include <vector>
#include <memory> // for std::unique_ptr and std::shared_ptr  <-- 注意这里
#include <limits> // for std::numeric_limits
#include <cmath>  // for std::sqrt
#include <atomic>

class Node
{
public:
    // 构造函数
    // 使用 std::move 高效地转移 game_state 的所有权
    Node(Node *parent = nullptr, int action_taken = -1, float prior = 0.0f);

    // vvv 将默认析构函数修改为自定义声明 vvv
    // ~Node() = default; // <-- 删除或注释掉这一行
    ~Node(); // <-- 新增这一行
    // ^^^ 修改结束 ^^^

    // 核心MCTS方法
    Node *select_child(float c_puct) const;
    // void expand(const std::vector<float>& policy);
    void backpropagate(double value);

    // 检查节点是否已经扩展过（即是否拥有子节点）
    bool is_fully_expanded() const;

    // 计算给定子节点的UCB值
    double get_ucb(const Node *child, float c_puct) const;

public: // 成员变量设为公有，方便C++ MCTS引擎直接访问，与Python版本保持一致
    Node *parent_;
    int action_taken_;
    float prior_;

    int visit_count_;
    double value_sum_;

    std::atomic<int> virtual_loss_count_;

    std::vector<Node *> children_; // 不再使用智能指针，改为原始指针

    mutable Gomoku *cached_state_ = nullptr;
    // std::shared_ptr<const Gomoku> game_state_;

    // MCTS超参数，从Python端传入
    // 为简化起见，我们可以在select_child中直接传递c_puct，而不是存储整个args字典
    // 这里我们直接定义 C (c_puct) 的值，在实际的MCTS引擎调用中可以传入此值
    // static constexpr float C_PUCT = 1.5f;
};