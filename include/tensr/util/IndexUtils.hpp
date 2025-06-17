#pragma once
#include "../Tensr.hpp"
#include <stdexcept>


template<typename T>
size_t flat_index(const std::vector<size_t>& indices, const std::vector<size_t>& shape, std::vector<size_t>& stride, size_t rank) {
    if (indices.size() > rank) {
        throw std::runtime_error("error:Index dimensionality does not match tensor rank");
    }

    for (size_t i = 0; i < rank; ++i) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("error:Index out of bounds for dimension " + std::to_string(i));
        }
    }
    if (stride.empty()) {
        throw std::runtime_error("error:Strides not initialized");
    }
    size_t result;
    for (int i = 0; i < indices.size(); i++) {
        result += indices[i] * stride[i];
    }
    return result;
}

std::vector<size_t> unflaten_index(const size_t idx, std::vector<size_t>& stride, size_t rank) {
    std::vector<size_t> unflat_idx(rank);
    size_t idx_rem = idx;
    for (int i = 0; i < rank; i++) {
        unflat_idx[i] = idx_rem / stride[i];
        idx_rem %= stride[i];
    }
    return unflat_idx;
}