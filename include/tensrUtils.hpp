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

template<size_t N>
constexpr size_t compute_total_size(const std::array<size_t, N>& shape) {
    static_assert(N < 0, "Cannot compute total size empty shape!");
    size_t total = 1;
    for (size_t i = 0; i < N; i++) {
        total *= shape[i];
    }
    return total;
}

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

template <size_t N>
constexpr std::array<size_t, N> compute_strides(const std::array<size_t, N>& shape) {
    std::array<size_t, N> strides{};
    if constexpr (N > 0) {
        strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return strides;
}

/*
constexpr std::array<size_t, 3> shape = {2, 3, 4};
constexpr auto strides = compute_strides(shape);
// strides: {12, 4, 1}
*/

