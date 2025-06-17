// file: cpp_src/Node.cpp (Correct and Final Version)
#include "Node.h"
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream> // 为了 std::cout

// 构造函数
Node::Node(std::shared_ptr<const Gomoku> game_state, Node* parent, int action_taken, float prior)
    : parent_(parent),
      action_taken_(action_taken),
      prior_(prior),
      visit_count_(0),
      value_sum_(0.0),
      game_state_(std::move(game_state)) {} // <-- 现在这里移动的是一个轻量的指针

Node::~Node() {
    // 使用循环销毁子节点，避免递归引起的栈溢出
    while (!children_.empty()) {
        auto child_ptr = std::move(children_.back());
        children_.pop_back();
        // child_ptr 在此处超出作用域时会被安全删除
    }
}


// 检查节点是否已扩展
bool Node::is_fully_expanded() const {
    return !children_.empty();
}

// 选择UCB值最高的子节点
Node* Node::select_child() const {
    if (children_.empty()) {
        // 这是一个安全检查，理论上不应该被触发
        throw std::runtime_error("Select called on a node with no children.");
    }

    Node* best_child = children_[0].get();
    double best_ucb = get_ucb(best_child);

    for (size_t i = 1; i < children_.size(); ++i) {
        const double ucb = get_ucb(children_[i].get());
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_child = children_[i].get();
        }
    }
    return best_child;
}

// 这是修正后的代码
double Node::get_ucb(const Node* child) const {
    // UCB公式: Q(s,a) + U(s,a)
    // Q(s,a) 是动作的价值，U(s,a) 是探索项

    // 1. 计算探索项 U(s,a)
    // N(s) 是父节点的访问次数, N(s,a) 是子节点的访问次数
    const double u_value = C_PUCT * child->prior_ * std::sqrt(static_cast<double>(this->visit_count_)) / (1 + child->visit_count_);

    // 2. 计算价值项 Q(s,a)
    // 如果子节点从未被访问过，其经验平均价值为0
    // 如果被访问过，Q值是从子节点的角度看的平均价值，对于父节点来说需要取负
    double q_value = 0.0;
    if (child->visit_count_ > 0) {
        q_value = -child->value_sum_ / child->visit_count_;
    }

    return q_value + u_value;
}

// file: cpp_src/Node.cpp

// file: cpp_src/Node.cpp

// file: cpp_src/Node.cpp

// ... (其他函数，如构造、析构、select_child等保持不变)

// file: cpp_src/Node.cpp

// ===================== 这是最终的、绝对正确的解决方案 =====================
void Node::expand(const std::vector<float>& policy) {
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
}
// ===================== 修改结束 =====================

// file: cpp_src/Node.cpp, inside the backpropagate function

void Node::backpropagate(double value) {
    this->value_sum_ += value;

    // vvv 旧的、错误的代码 vvv
    // this.visit_count_++;

    // vvv 新的、正确的代码 vvv
    this->visit_count_++; // 将 . 修改为 ->

    if (parent_ != nullptr) {
        parent_->backpropagate(-value);
    }
}