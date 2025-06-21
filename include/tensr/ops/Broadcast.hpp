#pragma once

#include "../Tensr.hpp"
#include "../Lens.hpp"
#include <vector>
#include <stdexcept>
#include <assert.h>
#include <stdio.h>

bool can_broadcast(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    const size_t max_rank = std::max(shape1.size(), shape2.size());

    for (size_t i = 0; i < max_rank; ++i) {
        size_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }

    return true;
}

std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2){
    if (can_broadcast(shape1, shape2)) {
        size_t result_rank = std::max(shape1.size(), shape2.size());
        std::vector<size_t> result_shape(result_rank);

        for (size_t i = 0; i < result_rank; i++) {
            size_t itl = (i < result_rank - shape1.size()) ? 1 : shape1[i - (result_rank - shape1.size())];
            size_t itr = (i < result_rank - shape2.size()) ? 1 : shape2[i - (result_rank - shape2.size())];

            if (itl == itr || itl == 1 || itr == 1) {
                result_shape[i] = std::max(itl, itr);
            } else {
                throw std::runtime_error("error: Shapes cannot be broadcasted");
            }
        }

        return result_shape;
    } else {
        throw std::runtime_error("Cannot broadcast");
    }
}

std::vector<size_t> compute_broadcast_stride(const std::vector<size_t>& orig_shape, const std::vector<size_t>& orig_stride, const std::vector<size_t>& target_shape) {
    assert(orig_stride.size() == orig_shape.size());

    std::vector<size_t> computed_stride;
    computed_stride.reserve(target_shape.size());
    auto ito = orig_shape.rbegin(), itt = target_shape.rbegin(), its = orig_stride.rbegin();
    while (ito != orig_shape.rend() && itt != target_shape.rend()) {
        if (*ito == *itt) {
            computed_stride.push_back(*its);
        } else if(*ito == 1 && *itt > 1) {
            computed_stride.push_back(0);
        } else {
            throw std::runtime_error("Unexpected Shape value");
        }


        ++ito, ++itt, ++its;
    }

    while (itt != target_shape.rend()) {
        computed_stride.push_back(0);
        ++itt;
    }
    std::reverse(computed_stride.begin(), computed_stride.end());
    return computed_stride;
}