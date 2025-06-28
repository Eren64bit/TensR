#pragma once


#include <stdexcept>
#include <numeric>
#include <vector>

inline size_t compute_rank(const std::vector<size_t>& shape) {
    return shape.size();
}

inline size_t compute_total_size(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Cannot compute total size empty shape!");
    }
    return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}//

inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return {};
    }
    
    std::vector<size_t> strides(shape.size());
    strides.back() = 1; 
    

    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    return strides;
}//


