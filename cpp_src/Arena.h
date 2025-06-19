#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <functional> // 为了 std::function

class Arena {
public:
    // 构造时分配一大块内存
    explicit Arena(size_t size_in_bytes) {
        buffer_ = std::make_unique<char[]>(size_in_bytes);
        capacity_ = size_in_bytes;
        offset_ = 0;
    }

    // Arena被销毁时，自动调用reset来清理所有对象
    ~Arena() {
        reset();
    }

    // 移动构造函数
    Arena(Arena&& other) noexcept
        : buffer_(std::move(other.buffer_)),
          capacity_(other.capacity_),
          offset_(other.offset_),
          // 一并移动析构回调
          destructor_callbacks_(std::move(other.destructor_callbacks_))
    {
        other.capacity_ = 0;
        other.offset_ = 0;
    }

    // 禁用拷贝
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // 分配内存的核心函数，现在它会记录如何销毁对象
    template <typename T>
    T* allocate(size_t count = 1) {
        // 对齐处理
        size_t align = alignof(T);
        size_t padding = (align - (offset_ % align)) % align;

        if (offset_ + padding + sizeof(T) * count > capacity_) {
            throw std::bad_alloc();
        }
        offset_ += padding;
        char* ptr = buffer_.get() + offset_;
        offset_ += sizeof(T) * count;

        // **核心改动**：为这个新创建的对象存储一个“析构回调函数”
        // 这个lambda函数捕获了类型T，并知道如何调用T的析构函数
        destructor_callbacks_.emplace_back([ptr, count] {
            T* obj_ptr = reinterpret_cast<T*>(ptr);
            for (size_t i = 0; i < count; ++i) {
                (obj_ptr + i)->~T();
            }
        });

        return reinterpret_cast<T*>(ptr);
    }

    // 将竞技场重置，以便重新使用
    void reset() {
        // **核心改动**：在重置指针之前，反向调用所有已记录的析构函数
        // 反向调用是为了保证销毁顺序与构造顺序相反，符合常规
        for (auto it = destructor_callbacks_.rbegin(); it != destructor_callbacks_.rend(); ++it) {
            (*it)();
        }
        // 清空回调记录并重置偏移
        destructor_callbacks_.clear();
        offset_ = 0;
    }

private:
    std::unique_ptr<char[]> buffer_;
    size_t capacity_;
    size_t offset_;

    // **核心改动**：使用 std::function 存储可以调用析构函数的闭包
    std::vector<std::function<void()>> destructor_callbacks_;
};