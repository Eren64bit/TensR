#pragma once

#define PRINT_THRESHOLD 6

#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <unordered_set>
#include <iostream>

namespace tensr_utils
{
  // Utility functions for tensor operations can be defined here.
  template <typename T>
  class fill_data
  {
  public:
    static void zeros(T *data, size_t size)
    {
      if (data == nullptr)
        throw std::invalid_argument("Data pointer must not be null.");
      for (size_t i = 0; i < size; ++i)
      {
        data[i] = T(0);
      }
    }

    static void custom(T *data, size_t size, const T &value)
    {
      if (data == nullptr)
        throw std::invalid_argument("Data pointer must not be null.");
      for (size_t i = 0; i < size; ++i)
      {
        data[i] = value;
      }
    }
  };
  // Function to compute strides and size from shape
  static std::vector<size_t> compute_strides(const std::vector<size_t> &shape)
  {
    std::vector<size_t> strides(shape.size());
    if (!shape.empty())
    {
      strides.back() = 1;
      for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
      {
        strides[i] = strides[i + 1] * shape[i + 1];
      }
    }
    return strides;
  }

  // Function to compute the total size of a tensor from its shape
  static size_t compute_size(const std::vector<size_t> &shape)
  {
    if (shape.empty())
      return 0;
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                           std::multiplies<size_t>());
  }

  // Function to compute the flattened index from shape, strides, and indices
  static size_t flatten_index(const std::vector<size_t> &shape,
                              const std::vector<size_t> &strides,
                              const std::vector<size_t> &indices)
  {
    if (shape.size() != indices.size() || shape.size() != strides.size())
    {
      throw std::invalid_argument(
          "Shape, strides, and indices must have the same size.");
    }
    size_t index = 0;
    for (size_t i = 0; i < shape.size(); ++i)
    {
      if (indices[i] >= shape[i])
      {
        throw std::out_of_range("Index out of bounds for the given shape.");
      }
      index += indices[i] * strides[i];
    }
    return index;
  }
  // Overloaded function to compute the flattened index using only shape and
  // indices
  static size_t flatten_index(const std::vector<size_t> &shape,
                              const std::vector<size_t> &indices)
  {
    std::vector<size_t> strides = compute_strides(shape);
    return flatten_index(shape, strides, indices);
  }

  static std::vector<size_t> unflatten_index(const std::vector<size_t> &shape,
                                             size_t index)
  {
    std::vector<size_t> strides = compute_strides(shape);
    std::vector<size_t> indices(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
      indices[i] = index / strides[i];
      index %= strides[i];
    }
    return indices;
  }
  //{3,4,5}
  template <typename T>
  static void print_data(const std::vector<size_t> &shape,
                         const std::shared_ptr<T[]> &data)
  {
    size_t total = 1;
    for (auto s : shape)
      total *= s;

    std::vector<size_t> strides = compute_strides(shape);

    for (size_t idx = 0; idx < total; ++idx)
    {
      std::vector<size_t> indices = unflatten_index(shape, idx);

      bool skip = false;
      for (size_t dim = 0; dim < shape.size(); ++dim)
      {
        if (shape[dim] > PRINT_THRESHOLD &&
            (indices[dim] != 0 && indices[dim] != shape[dim] - 1))
        {
          skip = true;
          break;
        }
      }

      if (!skip)
      {
        std::cout << "[";
        for (size_t i = 0; i < indices.size(); ++i)
        {
          std::cout << indices[i];
          if (i + 1 < indices.size())
            std::cout << ",";
        }
        std::cout << "] = " << data[idx] << "\n";
      }
      else if (idx % strides.back() == 0)
      {
        std::cout << "... (skipped)\n";
      }
    }
  }

} // namespace tensr_utils
