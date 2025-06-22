#pragma once

#include "../Tensr.hpp"
#include "../Lens.hpp"
#include "Broadcast.hpp"
#include "../util/TensrUtils.hpp"
#include "../util/IndexUtils.hpp"

template<typename T, typename TensorA, typename TensorB, typename BinaryOp>
Tensr<T> binary_op(const TensorA& t1, const TensorB& t2, BinaryOp op) {
    std::vector<size_t> common_shape = compute_broadcast_shape(t1.shape(), t2.shape());
    Tensr<T> result(common_shape);

    size_t total_size = compute_total_size(common_shape);
    std::vector<size_t> result_stride = compute_strides(common_shape);

    TensrLens<T> lens1 = (t1.shape() == common_shape) ? 
        TensrLens<T>(t1.data(), t1.shape(), t1.stride(), 0) :
        broadcast_to(t1, common_shape);

    TensrLens<T> lens2 = (t2.shape() == common_shape) ? 
        TensrLens<T>(t2.data(), t2.shape(), t2.stride(), 0) :
        broadcast_to(t2, common_shape);

    auto d1 = lens1.data().lock();
    auto d2 = lens2.data().lock();
    auto res_data = result.data();

    for (size_t idx = 0; idx < total_size; ++idx) {
        auto multi_idx = unflatten_index(idx, common_shape);
        size_t idx1 = flatten_index(multi_idx, lens1.stride(), lens1.offset());
        size_t idx2 = flatten_index(multi_idx, lens2.stride(), lens2.offset());

        (*res_data)[idx] = op((*d1)[idx1], (*d2)[idx2]);
    }

    return result;
}
