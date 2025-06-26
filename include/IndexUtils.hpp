#pragma once

#include <vector>
#include "tensrUtils.hpp"

namespace indexUtils {

inline size_t flat_index(const std::vector<size_t>& indices, const std::vector<size_t>& shape, const std::vector<size_t>& strides) {
    if (indices.size() > shape.size()) {
        throw std::runtime_error("error:Index dimensionality does not match tensor rank");
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("error:Index out of bounds for dimension " + std::to_string(i));
        }
    }

    if (strides.empty()) {
        throw std::runtime_error("error:Strides not initialized");
    }

    size_t result = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        result += indices[i] * strides[i];
    }
    return result;
}


inline std::vector<size_t> unflatten_index(const size_t index, const std::vector<size_t>& shape) {
    std::vector<size_t> res_vector(shape.size());
    std::vector<size_t> strides = compute_strides(shape);

    size_t idx_rem = index;
    for (int i = 0; i < strides.size(); i++) {
        res_vector[i] = idx_rem / strides[i];
        idx_rem %= strides[i];
    }

    return res_vector;
}

inline int normalize_index(int idx, size_t dim_size) {
    if (idx < 0) idx = static_cast<int>(dim_size) + idx;
    if (idx < 0 || idx >= static_cast<int>(dim_size)) {
        throw std::out_of_range("Normalized index out of bounds");
    }
    return idx;
}

}