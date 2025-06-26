#pragma once

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include "tensrUtils.hpp"
#include "tensrLens.hpp"

namespace broadcast {
    
    inline bool can_broadcast(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b) { // check for valid broadcast 
        const size_t max_rank = std::max(shape_a.size(), shape_b.size());

        for (size_t i = 0; i < max_rank; i++) {
            size_t dim_a = (i < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
            size_t dim_b = (i < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;

            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                return false;
            }
        }
        return true;
    }

    std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b) {
        if (can_broadcast(shape_a, shape_b)) {
            size_t result_rank = std::max(shape_a.size(), shape_b.size());
            std::vector<size_t> result_shape(result_rank);

            for (size_t i = 0; i < result_rank ; i++) {
                size_t dim_a = (i < result_rank - shape_a.size()) ? 1 : shape_a[i - (result_rank - shape_a.size())];
                size_t dim_b = (i < result_rank - shape_b.size()) ? 1 : shape_b[i - (result_rank - shape_b.size())];

                if (dim_a == dim_b || dim_a == 1 || dim_b == 1) {
                    result_shape[i] = std::max(dim_a, dim_b);
                } else {
                    throw std::runtime_error("error: Invalid shape alignment during stride computation");
                }
            } 

            return result_shape;
        } else {
            throw std::runtime_error("error: Invalid shape alignment during stride computation");
        }
    }

    std::vector<size_t> compute_broadcast_stride(const std::vector<size_t>& orig_shape, const std::vector<size_t>& target_shape) {
        const std::vector<size_t> orig_stride = compute_strides(orig_shape);

        std::vector<size_t> result_stride;
        result_stride.reserve(target_shape.size());

        auto ito = orig_shape.rbegin(), itt = target_shape.rbegin(), its = orig_stride.rbegin();
        while (ito != orig_shape.rend() && itt != target_shape.rend()) {
            if (*ito == *itt) {
                result_stride.push_back(*its);
            } else if (*ito == 1 && *itt > 1) {
                result_stride.push_back(0);
            } else {
                throw std::runtime_error("Unexpected Shape value");
            }
            ++ito, ++itt, ++its;
        }
        while (itt != target_shape.rend()) {
            result_stride.push_back(0);
            ++itt;
        }

        std::reverse(result_stride.begin(), result_stride.end());
        return result_stride;
        
    }


    template<typename T>
    tensrLens::lens<T> broadcast_to(const tensr::Tensr<T>& tensor, const std::vector<size_t>& target_shape) { // tensor overload
        std::vector<size_t> common_shape = compute_broadcast_shape(tensor.shape(), target_shape);
        std::vector<size_t> common_stride = compute_broadcast_stride(tensor.shape(), target_shape);

        tensrLens::lens<T> lens(tensor.data(), common_shape, common_stride, tensor.offset());
        return lens;
    }

    template<typename T>
    tensrLens::lens<T> broadcast_to(const tensrLens::lens<T>& lens, const std::vector<size_t>& target_shape) { // lens overload
        std::vector<size_t> common_shape = compute_broadcast_shape(lens.shape(), target_shape);
        std::vector<size_t> common_stride = compute_broadcast_stride(lens.shape(), target_shape);

        tensrLens::lens<T> comp_lens(lens.data(), common_shape, common_stride, lens.offset());
        return comp_lens;
    }



}