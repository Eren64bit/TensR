#pragma once

#include "allocator.hpp"
#include "tensr_interface.hpp"

template <typename T>
class tensr_view; // Forward declaration

// TensorStatic<T> represents a fixed-shape, fixed-type tensor.
// It is an abstract base class that provides a common interface for tensors
// with static shapes and types. Derived classes must implement the data()
// method

template <typename T, typename allocator = default_smart_allocator<T>>
class tensr_static : public tensr_interface<T>
{
protected:
  std::shared_ptr<T[]> data_;
  std::unique_ptr<allocator> alloc_;

  static_assert(std::is_arithmetic_v<T>,
                "error:Tensor type must be arithmetic");

public:
  explicit tensr_static(const std::vector<size_t> &shape, size_t offset = 0)
      : metadata_(shape, offset), alloc_(std::make_unique<allocator>())
  {
    data_ = alloc_->allocate(metadata_.size());
  }

  explicit tensr_static(const std::vector<size_t> &shape, const T data[],
                        size_t offset = 0)
      : metadata_(shape, offset), alloc_(std::make_unique<allocator>())
  {
    data_ = alloc_->allocate(metadata_.size());
    for (size_t i = 0; i < metadata_.size(); ++i)
    {
      data_[i] = data[i];
    }
  }

  // Data API
  [[nodiscard]] std::shared_ptr<T[]> data_owner() override
  {
    if (!data_)
      throw std::runtime_error("Data is not initialized.");
    return data_;
  }

  [[nodiscard]] const std::shared_ptr<T[]> data_owner() const override
  {
    if (!data_)
      throw std::runtime_error("Data is not initialized.");
    return data_;
  }

  [[nodiscard]] T *raw_data() override{ return data_.get(); }

  [[nodiscard]] T const *raw_data() const override { return data_.get(); }

  // Getter API
  [[nodiscard]] const std::vector<size_t> &shape() const 
  {
    return metadata_.shape();
  }
  [[nodiscard]] const std::vector<size_t> &strides() const 
  {
    return metadata_.strides();
  }

  [[nodiscard]] size_t size() const { return metadata_.size(); }
  [[nodiscard]] size_t rank() const { return metadata_.shape().size(); }

  // Access API
  // To do : add bound check to at functions
  T &at(const std::vector<size_t> &indices) override
  {
    if (indices.size() != metadata_.shape().size())
    {
      throw std::invalid_argument("Indices size must match shape size.");
    }
    size_t index = tensr_utils::flatten_index(metadata_.shape(),
                                              metadata_.strides(), indices);
    return data_[index + metadata_.offset()];
  }

  const T &at(const std::vector<size_t> &indices) const override
  {
    if (indices.size() != metadata_.shape().size())
    {
      throw std::invalid_argument("Indices size must match shape size.");
    }
    size_t index = tensr_utils::flatten_index(metadata_.shape(),
                                              metadata_.strides(), indices);
    return data_[index + metadata_.offset()];
  }

  T &operator()(const std::vector<size_t> &indices) override { return at(indices); }

  const T &operator()(const std::vector<size_t> &indices) const override
  {
    return at(indices);
  }

  T &operator[](size_t index) override
  {
    if (index >= metadata_.size())
    {
      throw std::out_of_range("Index out of bounds.");
    }
    return data_[index + metadata_.offset()];
  }

  const T &operator[](size_t index) const override
  {
    if (index >= metadata_.size())
    {
      throw std::out_of_range("Index out of bounds.");
    }
    return data_[index + metadata_.offset()];
  }

  // Fill API
  void fill(const T &value = T(0)) { fill_custom(value); }

  void fill_zeros()
  {
    if (!data_)
      throw std::runtime_error("Cannot fill: data is null.");
    tensr_utils::fill_data<T>::zeros(data_.get(), metadata_.size());
  }
  void fill_custom(const T &value)
  {
    if (!data_)
      throw std::runtime_error("Cannot fill: data is null.");
    tensr_utils::fill_data<T>::custom(data_.get(), metadata_.size(), value);
  }
};
