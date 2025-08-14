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

  tensr_metadata(const std::vector<size_t> &shape,
                 const std::vector<size_t> &strides,
                 size_t offset = 0)
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
                              size_t step = 1)
  {
    if (start.size() != source.rank() || stop.size() != source.rank())
      throw std::invalid_argument("slice function: start, stop size must match tensor rank");

    std::vector<size_t> new_shape(source.rank());
    std::vector<size_t> new_strides = source.strides();
    size_t new_offset = source.offset();

    for (size_t dim = 0; dim < source.rank(); ++dim)
    {
      if (start[dim] > stop[dim])
        throw std::invalid_argument("slice function: start must be <= stop for each dimension");
      if (step == 0)
        throw std::invalid_argument("slice function: step must be > 0");

      size_t length = stop[dim] - start[dim];
      size_t new_dim_size = (length + step - 1) / step;

      new_shape[dim] = (length + step - 1) / step;
      new_offset += start[dim] * source.strides()[dim];
      new_strides[dim] *= step;
    }

    return tensr_metadata(new_shape, new_strides, new_offset);
  }

  static tensr_metadata reshape(const tensr_metadata &source,
                                const std::vector<size_t> shape)
  {
    if (source.size() != tensr_utils::compute_size(shape))
      throw std::invalid_argument("reshape function: Metadata total size must match");
    return tensr_metadata(shape, tensr_utils::compute_strides(shape), source.offset());
  }

  static tensr_metadata transpose(const tensr_metadata &source,
                                  const std::vector<size_t> &perm = {})
  {
    std::vector<size_t> effective_perm;

    if (perm.empty())
    {
      effective_perm.resize(source.rank());
      std::iota(effective_perm.rbegin(), effective_perm.rend(), 0); // reverse order
    }
    else
    {
      if (perm.size() != source.rank())
        throw std::invalid_argument("transpose function: axes size must match tensor rank");
      effective_perm = perm;
    }

    if (std::unordered_set<size_t>(effective_perm.begin(), effective_perm.end()).size() != source.rank())
      throw std::invalid_argument("transpose function: permutation must contain all axes exactly once");

    for (size_t axes : effective_perm)
    {
      if (axes >= source.rank())
        throw std::invalid_argument("transpose function: invalid axes permutation");
    }

    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    for (size_t axes : effective_perm)
    {
      new_shape.push_back(source.shape()[axes]);
      new_strides.push_back(source.strides()[axes]);
    }

    return tensr_metadata(new_shape, new_strides, source.offset());
  }

  static tensr_metadata squeeze(const tensr_metadata &source, const size_t axis)
  {
    
    if (axis >= source.rank()) throw std::invalid_argument("squeeze function: axis out of range");
    if (source.shape()[axis] != 1) throw std::invalid_argument("squeeze function : cannot squeeze axis with size != 1");

    std::vector<size_t> new_shape;
    for (size_t i = 0; i <= source.rank(); ++i) 
    {
      if (i == axis) continue;
      new_shape.push_back(source.shape()[i]);
    }

    return tensr_metadata(new_shape, tensr_utils::compute_strides(new_shape), source.offset());
  }

  static tensr_metadata unsqueeze(const tensr_metadata &source, const size_t axis)
  {
    std::vector<size_t> new_shape;
    new_shape.reserve(source.rank() + 1);
    for (size_t i = 0; i < source.rank(); ++i)
    {
      if (i == axis)
        new_shape.push_back(1);
      if (i < source.rank())
      {
        new_shape.push_back(source.shape()[i]);
      }
    }

    return tensr_metadata(new_shape, tensr_utils::compute_strides(new_shape), source.offset());
  }
};
