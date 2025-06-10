// file: cpp_src/Node.cpp (Correct and Final Version)
#include "Node.h"
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cmath>

// 构造函数
Node::Node(Gomoku game_state, Node* parent, int action_taken, float prior)
    : parent_(parent),
      action_taken_(action_taken),
      prior_(prior),
      visit_count_(0),
      value_sum_(0.0),
      game_state_(std::move(game_state)) {}

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

// 计算UCB值
double Node::get_ucb(const Node* child) const {
    double q_value;
    if (child->visit_count_ == 0) {
        q_value = (this->visit_count_ > 0) ? -this->value_sum_ / this->visit_count_ : 0.0;
    } else {
        q_value = -child->value_sum_ / child->visit_count_;
    }
    const double u_value = C_PUCT * child->prior_ * std::sqrt(static_cast<double>(this->visit_count_)) / (1 + child->visit_count_);
    return q_value + u_value;
}

// 根据策略扩展节点
void Node::expand(const std::vector<float>& policy) {
    const auto valid_moves = game_state_.get_valid_moves();
    children_.reserve(policy.size());
    for (size_t action = 0; action < policy.size(); ++action) {
        if (policy[action] > 0.0f && valid_moves[action]) {
            auto child_game_ptr = std::make_unique<Gomoku>(game_state_);
            child_game_ptr->execute_move(action);
            children_.push_back(std::make_unique<Node>(std::move(*child_game_ptr), this, action, policy[action]));
        }
    }
}

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