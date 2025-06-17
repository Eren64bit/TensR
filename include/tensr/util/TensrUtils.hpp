#pragma once
#include <numeric>

#include "../Tensr.hpp"

template<typename T>
inline int compute_total_size(const std::vector<size_t> shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty"); //replace with Error class
    }
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>);
}

template<typename T>
inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

template<typename T>
inline int compute_rank(const std::vector<size_t>& shape) {
    return shape.size();
}
