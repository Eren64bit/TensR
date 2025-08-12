#pragma once
#include "tensr_utils.hpp"

class tensr_metadata
{
protected:
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  size_t offset_;
  size_t total_size_;
  bool is_contiguous_ = true;

public:
  tensr_metadata(const std::vector<size_t> &shape, size_t offset = 0)
      : shape_(shape), offset_(offset)
  {
    if (shape.empty())
      throw std::invalid_argument("Shape must not be empty.");
    strides_ = tensr_utils::compute_strides(shape_);
    total_size_ = size();
  }

  tensr_metadata(const std::vector<size_t> &shape, size_t offset,
                 const std::vector<size_t> &strides)
      : shape_(shape), strides_(strides), offset_(offset)
  {
    total_size_ = size();
  }

  [[nodiscard]] const std::vector<size_t> &shape() const { return shape_; }
  [[nodiscard]] const std::vector<size_t> &strides() const { return strides_; }

  [[nodiscard]] size_t size() const
  {
    return tensr_utils::compute_size(shape_);
  }
  [[nodiscard]] size_t rank() const { return shape_.size(); }

  [[nodiscard]] size_t offset() const { return offset_; }
  [[nodiscard]] size_t total_size() const { return total_size_; }

  void set_offset(size_t offset) { offset_ = offset; }

  size_t flatten_index(const std::vector<size_t> &indices) const
  {
    return tensr_utils::flatten_index(shape_, strides_, indices) + offset_;
  }

  // Shape API
  static tensr_metadata slice(const tensr_metadata &source,
                              const std::vector<size_t> &start,
                              const std::vector<size_t> &stop,
                              const std::vector<size_t> &step)
  {
    if (start.size() != source.rank() || stop.size() != source.rank() || step.size() != source.rank())
      throw std::invalid_argument("start, stop, step size must match tensor rank");

    tensr_metadata result = source;
    for (size_t dim = 0; dim < source.rank(); ++dim)
    {
      size_t length = (stop[dim] > start[dim]) ? (stop[dim] - start[dim]) : 0;
      size_t new_dim_size = (length + step[dim] - 1) / step[dim];

      result.shape();
    }
  }
};
