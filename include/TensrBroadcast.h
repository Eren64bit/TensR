#pragma once

#include "Tensr.h"
#include <functional>

template<typename T, DeviceType Device>
std::vector<size_t> broadcast_shapes(const Tensr<T, Device>& l_t, const Tensr<T, Device>& r_t) {
    const auto& l_shape = l_t.shape();
    const auto& r_shape = r_t.shape();

    size_t l_rank = l_t.rank();
    size_t r_rank = r_t.rank();

    size_t result_rank = std::max(l_rank, r_rank);

    std::vector<size_t> result_shape(result_rank, 1);

    for (size_t i = 0; i < result_rank; ++i) {
        size_t l_dim = (i < result_rank - l_rank) ? 1 : l_shape[i - (result_rank - l_rank)];
        size_t r_dim = (i < result_rank - r_rank) ? 1 : r_shape[i - (result_rank - r_rank)];

        if (l_dim == r_dim || l_dim == 1 || r_dim == 1) {
            result_shape[i] = std::max(l_dim, r_dim);
        } else {
            throw std::runtime_error("error: Shapes cannot be broadcasted");
        }
    }

    return result_shape;
}

template<typename T, DeviceType Device>
Tensr<T, Device> broadcast_to(Tensr<T, Device>& t, std::vector<size_t> result_shape) {
    std::vector<size_t> res_shape;
    for (size_t i = 0; i < result_shape.size() - t.rank(); i++) {
        res_shape.push_back(1);
    }   

    for (size_t i = 0; i < t.rank(); i++) {
        res_shape.push_back(t.shape()[i]);
    }

    Tensr<T, Device> res_tensor(res_shape);

    return res_tensor;
}

template<typename T, DeviceType Device>
Tensr<T, Device>broadcast_data(Tensr<T, Device>& t, std::vector<size_t>& res_shape) {
    for (size_t i = 0; i < t.shape()[0]; i++) {
        if (t.shape()[i] == 1) {
            for (size_t i = 0; i < res_shape[i] - t.shape()[i]; i++) {
                
            }
        } else {

        }
    }
}

// new shape [1,3] --> [4, 3] ----> before{1,2,3} after {0,0,0}{0,0,0}{0,0,0}{0,0,0}

template<typename T, DeviceType Device>
Tensr<T, Device> broadcast_binary_op(
    const Tensr<T, Device>& a,
    const Tensr<T, Device>& b,
    std::function<T(T, T)> op) 
{
    auto shape = broadcast_shapes(a, b);
    auto a_broad = broadcast_to(a, shape);
    auto b_broad = broadcast_to(b, shape);

    Tensr<T, Device> result(shape);
    auto& res_data = result.mutable_data();
    const auto& a_data = a_broad.data();
    const auto& b_data = b_broad.data();

    for (size_t i = 0; i < res_data.size(); i++) {
        res_data[i] = op(a_data[i], b_data[i]);
    }

    return result;
}   