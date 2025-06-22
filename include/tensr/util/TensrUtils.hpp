#pragma once
#include <numeric>

#include "../Tensr.hpp"
#include "../TensrTraits.hpp"
#include <algorithm>

inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape);

inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}


inline int compute_rank(const std::vector<size_t>& shape) {
    return shape.size();
}


inline size_t compute_total_size(const std::vector<size_t> shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty"); //replace with Error class
    }
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

template<typename TensorType>
void reshape(TensorType& tensor, const std::vector<size_t>& new_shape) {
    if (!TensrTraits<TensorType>::is_contiguous(tensor)) {
        throw std::runtime_error("Cannot reshape non-contiguous tensor");
    }

    TensrTraits<TensorType>::reshape(tensor, new_shape); 
}

template<typename TensorType>
void squeeze(TensorType& tensor) {
    if (TensrTraits<TensorType>::is_squeezable(tensor)) {
        TensrTraits<TensorType>::squeeze(tensor);
    } else {
        throw std::runtime_error("Cannot Squeeze");
    }
}

template<typename TensorType>
void squeeze(TensorType& tensor, int axis) {
    if (TensrTraits<TensorType>::is_squeezable(tensor)) {
        TensrTraits<TensorType>::squeeze(tensor, axis);
    } else {
        throw std::runtime_error("Cannot Squeeze");
    }
}

template<typename TensorType>
void unSqueeze(TensorType& tensor, int axis) {
    TensrTraits<TensorType>::unsqueeze(tensor, axis);
}