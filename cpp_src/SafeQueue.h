// file: cpp_src/SafeQueue.h
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

// 一个模板化的线程安全队列
template <typename T>
class SafeQueue {
public:
    // 向队列中推送一个元素
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_var_.notify_one(); // 通知一个等待的线程
    }

    // 从队列中弹出一个元素（如果队列为空，则阻塞等待）
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        // 当队列为空时，等待通知
        cond_var_.wait(lock, [this]{ return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    // 尝试从队列中弹出一个元素（如果队列为空，立即返回false）
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_; // 互斥锁，保护队列
    std::condition_variable cond_var_; // 条件变量，用于线程间的同步
};