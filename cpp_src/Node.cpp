// file: cpp_src/Node.cpp (Correct and Final Version)
#include "Node.h"
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream> // 为了 std::cout

// 构造函数
Node::Node(Node *parent, int action_taken, float prior)
    : parent_(parent),
      action_taken_(action_taken),
      prior_(prior),
      visit_count_(0),
      value_sum_(0.0),
      virtual_loss_count_(0) {} // 移除了 game_state_ 的初始化

Node::~Node()
{
    // 不需要做任何事情了！
    // 所有子节点的内存都由Arena统一管理和释放。
}

// 检查节点是否已扩展
bool Node::is_fully_expanded() const
{
    return !children_.empty();
}

// file: cpp_src/Node.cpp (修正后的版本)

Node *Node::select_child(float c_puct) const
{
    if (children_.empty())
    {
        throw std::runtime_error("Select called on a node with no children.");
    }

    Node *best_child = children_[0]; // <--- 直接使用，不再调用 .get()
    double best_ucb = get_ucb(best_child, c_puct);

    for (size_t i = 1; i < children_.size(); ++i)
    {
        // 直接将 children_[i] (它本身就是 Node*) 传递给 get_ucb
        const double ucb = get_ucb(children_[i], c_puct);
        if (ucb > best_ucb)
        {
            best_ucb = ucb;
            best_child = children_[i]; // <--- 直接赋值，不再调用 .get()
        }
    }
    return best_child;
}

// 在文件 cpp_src/Node.cpp 中，找到并替换整个 get_ucb 函数

double Node::get_ucb(const Node *child, float c_puct) const
{
    // vvvvvv 【核心修改】 vvvvvv
    // UCB公式现在需要同时考虑真实访问和虚拟损失

    // 探索项 U(s,a)
    // 分子中的父节点总“等效”访问次数 N(s)
    double parent_total_visits = static_cast<double>(this->visit_count_ + this->virtual_loss_count_);
    // 分母中的子节点总“等效”访问次数 N(s,a)
    double child_total_visits = static_cast<double>(child->visit_count_ + child->virtual_loss_count_);

    const double u_value = c_puct * child->prior_ * std::sqrt(parent_total_visits) / (1 + child_total_visits);

    // 价值项 Q(s,a)
    // 将每次虚拟访问视为一次值为-1的“损失”，这是虚拟损失的标准实现方法
    double q_value = 0.0;
    if (child_total_visits > 0)
    {
        // Q = (真实的价值总和 - 虚拟损失次数) / (真实访问次数 + 虚拟损失次数)
        // 注意：value_sum 已经是对手视角的，所以对于父节点是 -value_sum
        q_value = (child->value_sum_ - child->virtual_loss_count_) / child_total_visits;
    }

    // 对于父节点来说，子节点的Q值需要取反
    return -q_value + u_value;
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^
}

// file: cpp_src/Node.cpp

// file: cpp_src/Node.cpp

// file: cpp_src/Node.cpp

// ... (其他函数，如构造、析构、select_child等保持不变)

// file: cpp_src/Node.cpp

// ===================== 这是最终的、绝对正确的解决方案 =====================
// void Node::expand(const std::vector<float>& policy) {
/*const auto valid_moves = game_state_.get_valid_moves();
children_.reserve(policy.size());

for (size_t action = 0; action < policy.size(); ++action) {
    if (policy[action] > 0.0f && valid_moves[action]) {
        // 关键修正：在循环的每一步都从 "this->game_state_" 这个原始、干净的状态出发，
        // 创建一个全新的副本。
        Gomoku next_game_state = this->game_state_;

        // 在这个全新的副本上安全地执行一步
        next_game_state.execute_move(action);

        // 将这个副本传递给子节点。由于参数是按值传递的，
        // C++会自动为构造函数创建一个临时拷贝，然后这个拷贝被 move 进子节点，
        // 这不会影响我们下一次循环时创建的 next_game_state。
        // 为了更清晰，我们也可以直接传递一个拷贝。
        children_.push_back(std::make_unique<Node>(next_game_state, this, action, policy[action]));
    }
}*/
//}
// ===================== 修改结束 =====================

// 在文件 cpp_src/Node.cpp 中，找到 backpropagate 函数并修改

void Node::backpropagate(double value)
{
    // vvvvvv 【核心修改】 vvvvvv
    // 在反向传播时，一个“虚拟损失”被一次“真实访问”所替代
    this->virtual_loss_count_--; // 偿还（撤销）一次虚拟损失
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    this->visit_count_++; // 增加一次真实访问
    this->value_sum_ += value;

    if (parent_ != nullptr)
    {
        parent_->backpropagate(-value);
    }
}