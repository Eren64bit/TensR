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
Tensr<T, Device> broadcast_to(const Tensr<T, Device>& src, const std::vector<size_t>& target_shape) {
    Tensr<T, Device> dst(target_shape);          
    broadcast_data(src, dst);                    
    return dst;
 }

template<typename T, DeviceType Device>
void broadcast_data(Tensr<T, Device>& source, Tensr<T, Device>& target) {
    for (size_t flat_idx = 0; flat_idx < target.size(); flat_idx++) {
        std::vector<size_t> multi_idx = target.unflaten_index_(flat_idx);

        size_t offset = multi_idx.size() - source.rank();

        std::vector<size_t> orig_idx;
        for (size_t i = 0; i < multi_idx.size(); i++) {
            if (i < offset) {
                orig_idx.push_back(0);
            } else if (source.shape()[i - offset] == 1){
                orig_idx.push_back(0);
            } else {
                orig_idx.push_back(multi_idx[i]);
            }
                
        }

        target.mutable_data()[flat_idx] = source.at(orig_idx);
    }
}

template<typename T, DeviceType Device>
Tensr<T, Device> broadcast_binary_op(Tensr<T, Device>& lhs, Tensr<T, Device>& rhs, std::function<T(T, T)> func) {
    std::vector<size_t> cmmn_shape = broadcast_shapes(lhs, rhs);
    Tensr<T, Device> result(cmmn_shape);
    Tensr<T, Device> br_lhs = lhs;
    Tensr<T, Device> br_rhs = rhs;
    if (lhs.shape() != cmmn_shape) {
        br_lhs = broadcast_to(lhs, cmmn_shape);
        broadcast_data(lhs, br_lhs);
    }
    if (rhs.shape() != cmmn_shape) {
        br_rhs = broadcast_to(rhs, cmmn_shape);
        broadcast_data(rhs, br_rhs);
    }

    for (int i = 0; i < result.size(); i++) {
        result[i] = func(br_lhs[i], br_rhs[i]);
    }

    return result;
}
