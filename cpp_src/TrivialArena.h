#pragma once

#include <memory>
#include <stdexcept>
#include <cstddef> // 为了 std::size_t 和 std::alignof

// 一个模板化的、用于管理“平凡可析构”类型对象的内存竞技场。
// 它的 reset() 方法是 O(1) 复杂度。
class TrivialArena
{
public:
    // 构造时分配一大块内存
    explicit TrivialArena(size_t size_in_bytes)
    {
        buffer_ = std::make_unique<char[]>(size_in_bytes);
        capacity_ = size_in_bytes;
        offset_ = 0;
    }

    // 禁用拷贝构造和赋值
    TrivialArena(const TrivialArena &) = delete;
    TrivialArena &operator=(const TrivialArena &) = delete;

    // 允许移动构造，以方便在类成员间转移所有权
    TrivialArena(TrivialArena &&other) noexcept
        : buffer_(std::move(other.buffer_)),
          capacity_(other.capacity_),
          offset_(other.offset_)
    {
        other.capacity_ = 0;
        other.offset_ = 0;
    }

    // 分配内存的核心函数
    template <typename T>
    T *allocate(size_t count = 1)
    {
        // 对齐处理
        size_t align = alignof(T);
        size_t padding = (offset_ % align == 0) ? 0 : (align - (offset_ % align));

        if (offset_ + padding + sizeof(T) * count > capacity_)
        {
            throw std::bad_alloc();
        }

        offset_ += padding;
        char *ptr = buffer_.get() + offset_;
        offset_ += sizeof(T) * count;

        return reinterpret_cast<T *>(ptr);
    }

    // O(1) 复杂度的重置方法，极致高效
    void reset()
    {
        offset_ = 0;
    }

    // 获取当前容量
    size_t capacity() const
    {
        return capacity_;
    }

private:
    std::unique_ptr<char[]> buffer_;
    size_t capacity_;
    size_t offset_;
};