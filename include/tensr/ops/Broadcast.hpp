#pragma once

#include "../Tensr.hpp"
#include "../Lens.hpp"
#include <vector>
#include <stdexcept>


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