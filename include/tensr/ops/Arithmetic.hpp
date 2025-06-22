#pragma once
#include "../Tensr.hpp"
#include "../Lens.hpp"
#include "Broadcast.hpp"
#include "../util/TensrUtils.hpp"
#include "../util/IndexUtils.hpp"

template<typename T, typename TensorA, typename TensorB, typename BinaryOp>
Tensr<T> binary_op(const TensorA& t1, const TensorB& t2, BinaryOp op) {
    // Early exit for same shape + contiguous case
    if (t1.shape() == t2.shape() && t1.is_contiguous() && t2.is_contiguous()) {
        Tensr<T> result(t1.shape());
        size_t total_size = compute_total_size(t1.shape());
        auto d1 = t1.data();
        auto d2 = t2.data();
        auto res_data = result.data();
        
        for (size_t i = 0; i < total_size; ++i) {
            (*res_data)[i] = op((*d1)[i], (*d2)[i]);
        }
        return result;
    }
    
    // General broadcasting case
    std::vector<size_t> common_shape = compute_broadcast_shape(t1.shape(), t2.shape());
    Tensr<T> result(common_shape);
    size_t total_size = compute_total_size(common_shape);
    
    TensrLens<T> lens1 = (t1.shape() == common_shape) ?
        TensrLens<T>(t1.data(), t1.shape(), t1.stride(), 0) :
        broadcast_to(t1, common_shape);
    
    TensrLens<T> lens2 = (t2.shape() == common_shape) ?
        TensrLens<T>(t2.data(), t2.shape(), t2.stride(), 0) :
        broadcast_to(t2, common_shape);
    
    auto d1 = lens1.data().lock();
    auto d2 = lens2.data().lock();
    auto res_data = result.data();
    
    // Use compute_broadcast_stride for optimized stride calculation
    std::vector<size_t> stride1_mapped = compute_broadcast_stride(
        lens1.shape(), lens1.stride(), common_shape);
    std::vector<size_t> stride2_mapped = compute_broadcast_stride(
        lens2.shape(), lens2.stride(), common_shape);
    
    // Direct stride calculation loop
    for (size_t flat_idx = 0; flat_idx < total_size; ++flat_idx) {
        size_t idx1 = lens1.offset();
        size_t idx2 = lens2.offset();
        size_t temp = flat_idx;
        
        // Convert flat index to multi-dimensional coordinates and apply strides
        for (int dim = static_cast<int>(common_shape.size()) - 1; dim >= 0; --dim) {
            size_t coord = temp % common_shape[dim];
            temp /= common_shape[dim];
            idx1 += coord * stride1_mapped[dim];
            idx2 += coord * stride2_mapped[dim];
        }
        
        (*res_data)[flat_idx] = op((*d1)[idx1], (*d2)[idx2]);
    }
    
    return result;
}