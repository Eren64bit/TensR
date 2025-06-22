#pragma once

#include "../Tensr.hpp"
#include "../Lens.hpp"
#include "Broadcast.hpp"
#include "../util/TensrUtils.hpp"
#include "../util/IndexUtils.hpp"

template<typename T,typename TensorA, typename TensorB, typename BinaryOp>
Tensr<T> binary_op(const TensorA& t1, const TensorB& t2, BinaryOp op) {
    std::vector<size_t> common_shape = compute_broadcast_shape(t1.shape(), t2.shape());
    Tensr<T> result_tensr(common_shape);
    std::vector<size_t> res_stride = compute_strides(common_shape);

    TensrLens<T> lens1;
    TensrLens<T> lens2;

    if (t1.shape() == t2.shape()) {
        lens1.set_data(t1.data());
        lens1.set_shape(t1.shape());
        lens1.set_stride(t1.stride());
        lens1.set_offset((size_t)0);

        lens2.set_data(t2.data());
        lens2.set_shape(t2.shape());
        lens2.set_stride(t2.stride());
        lens2.set_offset((size_t)0);
    } else {
        lens1 = broadcast_to(t1, common_shape);
        lens2 = broadcast_to(t2, common_shape);
    }

    for (size_t idx = 0; idx < compute_total_size(common_shape); idx++) {
        std::vector<size_t> multi_idx = unflaten_index(idx, res_stride, result_tensr.shape());

        auto a = lens1.at(multi_idx);
        auto b = lens2.at(multi_idx);

        result_tensr.at(multi_idx) = op(a, b);
    }

    return result_tensr;
}