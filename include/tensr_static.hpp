#pragma once

#include "allocator.hpp"
#include "tensr_view.hpp"

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
  tensr_metadata metadata_;

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

  [[nodiscard]] std::shared_ptr<T[]> data_owner() const override
  {
    if (!data_)
      throw std::runtime_error("Data is not initialized.");
    return data_;
  }

  [[nodiscard]] T *raw_data() override { return data_.get(); }

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

  // Shape API
  tensr_view<T> reshape(const std::vector<size_t> new_shape)
  {
    auto new_meta = tensr_metadata::reshape(this->metadata_, new_shape);
    return tensr_view<T>(new_meta, this->data_owner());
  }

  tensr_view<T> slice(const std::vector<size_t> &start, const std::vector<size_t> &stop, size_t step = 1)
  {
    auto new_meta = tensr_metadata::slice(this->metadata_, start, stop, step);
    return tensr_view<T>(new_meta, this->data_owner());
  }

  tensr_view<T> transpose(const std::vector<size_t> &perm = {})
  {
    auto new_meta = tensr_metadata::transpose(this->metadata_, perm);
    return tensr_view<T>(new_meta, this->data_owner());
  }

  tensr_view<T> squeeze(const size_t axis)
  {
    auto new_meta = tensr_metadata::squeeze(this->metadata_, axis);
    return tensr_view<T>(new_meta, this->data_owner());
  }

  tensr_view<T> unsqueeze(const size_t axis)
  {
    auto new_meta = tensr_metadata::unsqueeze(this->metadata_, axis);
    return tensr_view<T>(new_meta, this->data_owner());
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

  // Info Function
  void info(const std::string &name = "Tensor") const
  {
    std::cout << "=== " << name << " Metadata ===\n";
    std::cout << "Rank         : " << metadata_.rank() << "\n";
    std::cout << "Shape        : [";
    for (size_t i = 0; i < metadata_.rank(); ++i)
    {
      std::cout << metadata_.shape()[i];
      if (i + 1 != metadata_.shape().size())
        std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "Strides      : [";
    for (size_t i = 0; i < strides_.size(); ++i)
    {
      std::cout << metadata_.strides()[i];
      if (i + 1 != metadata_.strides().size())
        std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "Offset       : " << metadata_.offset() << "\n";
    std::cout << "Total Size   : " << metadata_.size() << "\n";
    std::cout << "==========================\n";
  }

  void visualize(const std::string &name = "Tensor Data:") const 
  {
    std::cout << name << " = " << std::endl;
    tensr_utils::print_data(metadata_.shape(), this->data_owner());
    
  }
};
