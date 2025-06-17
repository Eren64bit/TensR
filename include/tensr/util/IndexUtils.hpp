#pragma once
#include "../Tensr.hpp"
#include <stdexcept>


template<typename T>
size_t Flat_index(const std::vector<size_t>& indices, const std::vector<size_t>& shape, std::vector<size_t>& stride, size_t rank) {
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

std::vector<size_t> Unflaten_index(const std::vector<size_t>& indices, const std::vector<size_t>& shape, std::vector<size_t>& stride, size_t rank) {
    std::vector<size_t> unflat_idx(rank);
}