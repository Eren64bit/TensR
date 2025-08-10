#pragma once
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>

template <typename T> class allocator {
public:
  virtual T *allocate(size_t n) = 0;
  virtual void deallocate(T *p, size_t n) = 0;
  virtual ~allocator() = default;
};

template <typename T>
class default_allocator_performance : public allocator<T> {
public:
  T *allocate(size_t n) override {
    if (n == 0)
      return nullptr;
    T *p = static_cast<T *>(::operator new(n * sizeof(T)));
    if (!p)
      throw std::bad_alloc();
    return p;
  }

  void deallocate(T *p, size_t /*n*/) override { ::operator delete(p); }
};

template <typename T> class smart_allocator {
public:
  virtual std::shared_ptr<T[]> allocate(size_t count) = 0;
  virtual void deallocate(std::shared_ptr<T[]> ptr, size_t count) = 0;
  virtual ~smart_allocator() = default;
};

template <typename T>
class default_smart_allocator : public smart_allocator<T> {
public:
  std::shared_ptr<T[]> allocate(size_t count) override {
    return std::shared_ptr<T[]>(new T[count]);
  }
  void deallocate(std::shared_ptr<T[]> /*ptr*/, size_t /*count*/) override {}
};
