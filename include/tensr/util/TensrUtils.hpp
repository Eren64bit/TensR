#pragma once
#include <numeric>

#include "../Tensr.hpp"
#include "../TensrTraits.hpp"

template<typename T>
inline size_t compute_total_size(const std::vector<size_t> shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty"); //replace with Error class
    }
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>);
}


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


template<typename TensorType>
void reshape(TensorType& tensor, const std::vector<size_t>& new_shape) {
    if (! TensrTraits<TensorType>::is_contiguous(tensor)) {
        throw std::runtime_error("Cannot reshape non-contiguous tensor");
    }

    TensrTraits<TensorType>::reshape(tensor, new_shape); 
}