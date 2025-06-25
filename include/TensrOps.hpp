#pragma once

#include "tensrLens.hpp"
#include "tensr.hpp"
#include <bits/stdc++.h>


namespace tensrOps {

    template<typename T>
    tensrLens::lens<T> transpose(const tensr::Tensr<T>& tensor, const std::vector<size_t>& perm) {
        if (perm.size() != tensor.rank()) {
            throw std::invalid_argument("lens::transpose: axes size must match tensor rank");
        }

        std::vector<bool> used(tensor.rank(), false);
        for (size_t axis : perm) {
            if (axis >= tensor.rank() || used[axis]) {
                throw std::invalid_argument("lens::transpose: invalid axes permutation");
            }
            used[axis] = true;
        }

        std::vector<size_t> new_shape, new_stride;
        for (size_t axis : perm) {
            new_shape.push_back(tensor.shape()[axis]);
            new_stride.push_back(tensor.stride()[axis]);
        }

        return tensrLens::lens<T>(tensor.data(), new_shape, new_stride, tensor.offset());
    }

    template<typename T>
    tensrLens::lens<T> transpose(const tensr::Tensr<T>& tensor) {
        std::vector<size_t> perm(tensor.rank());
        std::iota(perm.begin(), perm.end(), 0);
        std::reverse(perm.begin(), perm.end());
        return transpose(tensor, perm)
    }


    template<typename T>
    tensrLens::lens<T> transpose(const tensrLens::lens<T>& lens, const std::vector<size_t>& perm) {
        if (perm.size() != lens.rank()) {
            throw std::invalid_argument("lens::transpose: axes size must match tensor rank");
        }

        std::vector<bool> used(lens.rank(), false);
        for (size_t axis : perm) {
            if (axis >= lens.rank() || used[axis]) {
                throw std::invalid_argument("lens::transpose: invalid axes permutation");
            }
            used[axis] = true;
        }

        std::vector<size_t> new_shape, new_stride;
        for (size_t axis : perm) {
            new_shape.push_back(lens.shape()[axis]);
            new_stride.push_back(lens.stride()[axis]);
        }

        return tensrLens::lens<T>(lens.data(), new_shape, new_stride, lens.offset());
    }

    template<typename T>
        tensrLens::lens<T> transpose(const tensrLens::lens<T>& lens) {
        std::vector<size_t> perm(lens.rank());
        std::iota(perm.begin(), perm.end(), 0);
        std::reverse(perm.begin(), perm.end());
        return transpose(lens, perm);
    }
}

