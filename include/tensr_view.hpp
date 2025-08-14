#pragma once

#include "tensr_metadata.hpp"
#include "tensr_interface.hpp"

template <typename T>
class tensr_view : public tensr_interface<T>
{
  std::shared_ptr<T[]> data_owner_;
  T *data_ptr_;

public:
  tensr_view(const tensr_metadata &view_meta,
             std::shared_ptr<T[]> data_ptr_owner)
      : tensr_interface<T>(view_meta), data_owner_(data_ptr_owner),
        data_ptr_(data_ptr_owner.get() + view_meta.offset()) {}

  tensr_view(const tensr_view &other) = default;

  tensr_view(tensr_view &&other) noexcept = default;

  tensr_view &operator=(const tensr_view &other) = default;

  tensr_view &operator=(tensr_view &&other) noexcept = default;

  ~tensr_view() = default;

  // Metadata access
  const tensr_metadata &metadata() const { return this->metadata_; }
  const std::vector<size_t> &shape() const { return this->metadata_.shape(); }
  size_t size() const { return this->metadata_.size(); }
  size_t rank() const { return this->metadata_.rank(); }

  // Data access
  T *raw_data() override { return data_ptr_; }
  const T *raw_data() const override { return data_ptr_; }
  std::shared_ptr<T[]> data_owner() override { return data_owner_; }
  std::shared_ptr<T[]> data_owner() const override { return data_owner_; }

  // Element access
  // To do : add bound check to at functions
  T &at(const std::vector<size_t> &indices) override
  {
    if (indices.size() != this->metadata_.rank())
    {
      throw std::invalid_argument("Indices size must match shape size.");
    }
    const auto &shape_m = this->metadata_.shape();
    const auto &strides_m = this->metadata_.strides();
    size_t index = tensr_utils::flatten_index(shape_m, strides_m, indices);
    return data_ptr_[index];
  }

  const T &at(const std::vector<size_t> &indices) const override
  {
    if (indices.size() != this->metadata_.shape().size())
    {
      throw std::invalid_argument("Indices size must match shape size.");
    }
    size_t index = tensr_utils::flatten_index(this->metadata_.shape(),
                                              this->metadata_.strides(), indices);
    return data_ptr_[index];
  }

  T &operator()(const std::vector<size_t> &indices) override { return at(indices); }

  const T &operator()(const std::vector<size_t> &indices) const override
  {
    return at(indices);
  }

  T &operator[](size_t index) override
  {
    if (index >= this->metadata_.size())
    {
      throw std::out_of_range("Index out of bounds.");
    }
    return data_ptr_[index];
  }

  const T &operator[](size_t index) const override
  {
    if (index >= this->metadata_.size())
    {
      throw std::out_of_range("Index out of bounds.");
    }
    return data_ptr_[index];
  }

  // Shape API
  tensr_view<T> reshape(const std::vector<size_t> new_shape)
  {
    auto new_meta = tensr_metadata::reshape(this->metadata_, new_shape);
    return tensr_view<T>(new_meta, this->data_owner());
  }

  tensr_view<T> slice(const std::vector<size_t> &start, const std::vector<size_t> &stop, size_t &step = 1)
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

  // Info Function
  void info(const std::string &name = "View") const
  {
    std::cout << "=== " << name << " Metadata ===\n";
    std::cout << "Rank         : " << this->metadata_.rank() << "\n";
    std::cout << "Shape        : [";
    for (size_t i = 0; i < this->metadata_.rank(); ++i)
    {
      std::cout << this->metadata_.shape()[i];
      if (i + 1 != this->metadata_.shape().size())
        std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "Strides      : [";
    for (size_t i = 0; i < this->metadata_.strides().size(); ++i)
    {
      std::cout << this->metadata_.strides()[i];
      if (i + 1 != this->metadata_.strides().size())
        std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "Offset       : " << this->metadata_.offset() << "\n";
    std::cout << "Total Size   : " << this->metadata_.size() << "\n";
    std::cout << "==========================\n";
  }

  void visualize(const std::string &name = "View Data:") const 
  {
    std::cout << name << " = " << std::endl;
    std::cout << "View does not own data." << std::endl;
    
  }
};
