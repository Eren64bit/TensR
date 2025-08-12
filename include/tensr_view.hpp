#pragma once

#include "tensr_static.hpp"

template <typename T>
class tensr_view : tensr_interface<T>
{
  std::shared_ptr<T[]> data_owner_;
  T *data_ptr_;

public:
  tensr_view(const tensr_metadata &view_meta,
             std::shared_ptr<T[]> data_ptr_owner)
      : metadata_(view_meta), data_owner_(data_ptr_owner),
        data_ptr_(data_ptr_owner.get() + view_meta.offset()) {}

  tensr_view(const tensr_view& other) = default;

  tensr_view(tensr_view&& other) noexcept = default;

  tensr_view& operator=(const tensr_view& other) = default;

  tensr_view& operator=(tensr_view&& other) noexcept = default;

  ~tensr_view() = default;
  

  // Metadata access
  const tensr_metadata &metadata() const { return metadata_; }
  const std::vector<size_t> &shape() const { return metadata_.shape(); }
  size_t size() const { return metadata_.size(); }
  size_t rank() const { return metadata_.rank(); }

  // Data access
  T *raw_data() override { return data_ptr_; }
  const T *raw_data() const override { return data_ptr_; }
  std::shared_ptr<T[]> data_owner() override { return data_owner_; }
  std::shared_ptr<T[]> const data_owner() const override { return data_owner_; }

  // Element access
  // To do : add bound check to at functions
  T &at(const std::vector<size_t> &indices) override
  {
    if (indices.size() != metadata_.shape().size())
    {
      throw std::invalid_argument("Indices size must match shape size.");
    }
    size_t index = tensr_utils::flatten_index(metadata_.shape(),
                                              metadata_.strides(), indices);
    return data_ptr_[index];
  }

  const T &at(const std::vector<size_t> &indices) const override
  {
    if (indices.size() != metadata_.shape().size())
    {
      throw std::invalid_argument("Indices size must match shape size.");
    }
    size_t index = tensr_utils::flatten_index(metadata_.shape(),
                                              metadata_.strides(), indices);
    return data_ptr_[index];
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
    return data_ptr_[index];
  }

  const T &operator[](size_t index) const override
  {
    if (index >= metadata_.size())
    {
      throw std::out_of_range("Index out of bounds.");
    }
    return data_ptr_[index];
  }

};
