#pragma once

#include "tensrLens.hpp"
#include "tensr.hpp"
#include "indexUtils.hpp"
#include <bits/stdc++.h>
#include "tensrOps_decl.hpp"


namespace tensrOps {
    //-----------------------------Transpose
    template<typename T>
    tensrLens::lens<T> transpose(const tensr::Tensr<T>& tensor, const std::vector<size_t>& perm) { // tensor overload
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


        return tensrLens::lens<T>(tensor.data().lock(), new_shape, new_stride, tensor.offset());
    }

    template<typename T>
    tensrLens::lens<T> transpose(const tensr::Tensr<T>& tensor) { // tensor reverse overload
        std::vector<size_t> perm(tensor.rank());
        std::iota(perm.begin(), perm.end(), 0);
        std::reverse(perm.begin(), perm.end());
        return transpose(tensor, perm);
    }


    template<typename T>
    tensrLens::lens<T> transpose(const tensrLens::lens<T>& lens, const std::vector<size_t>& perm) { // lens overload
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

        return tensrLens::lens<T>(lens.data().lock(), new_shape, new_stride, lens.offset());
    }

    template<typename T>
        tensrLens::lens<T> transpose(const tensrLens::lens<T>& lens) { // lens reverse overload
        std::vector<size_t> perm(lens.rank());
        std::iota(perm.begin(), perm.end(), 0);
        std::reverse(perm.begin(), perm.end());
        return transpose(lens, perm);
    }

    //----------------------------------Slice
    template<typename T>
    tensrLens::lens<T> slice(const tensr::Tensr<T>& tensor, const std::vector<SliceRange>& ranges) { // tensor overload
        const auto& orig_shape = tensor.shape();
        const auto& orig_stride = tensor.stride();

        if (ranges.size() != orig_shape.size()) {
            throw std::invalid_argument("Slice dimension count doesn't match tensor rank.");
        }

        std::vector<size_t> new_shape;
        std::vector<size_t> new_stride;
        size_t new_offset = tensor.offset();

        for (size_t i = 0; i < ranges.size(); ++i) {
            const auto& r = ranges[i];
            size_t dim = orig_shape[i];


            int start = indexUtils::normalize_index(r.start, dim);
            int stop = indexUtils::normalize_index(r.stop, dim);

            if (start < 0 || stop < 0 || start >= static_cast<int>(dim) || stop > static_cast<int>(dim) || r.step == 0) {
                throw std::out_of_range("Invalid (normalized) slice range");
            }

            size_t len = (stop - start + r.step - 1) / r.step;
            new_shape.push_back(len);
            new_stride.push_back(orig_stride[i] * r.step);
            new_offset += start * orig_stride[i];
        }

        return tensrLens::lens<T>(tensor.data().lock(), new_shape, new_stride, new_offset);
    }

    template<typename T>
    tensrLens::lens<T> slice(const tensrLens::lens<T>& lensW, const std::vector<SliceRange>& ranges) { // lens overload
        const auto& orig_shape = lensW.shape();
        const auto& orig_stride = lensW.stride();

        if (ranges.size() != orig_shape.size()) {
            throw std::invalid_argument("Slice dimension count doesn't match lens rank.");
        }

        std::vector<size_t> new_shape;
        std::vector<size_t> new_stride;
        size_t new_offset = lensW.offset();

       for (size_t i = 0; i < ranges.size(); ++i) {
            const auto& r = ranges[i];
            size_t dim = orig_shape[i];


            int start = indexUtils::normalize_index(r.start, dim);
            int stop = indexUtils::normalize_index(r.stop, dim);

            if (start < 0 || stop < 0 || start >= static_cast<int>(dim) || stop > static_cast<int>(dim) || r.step == 0) {
                throw std::out_of_range("Invalid (normalized) slice range");
            }

            size_t len = (stop - start + r.step - 1) / r.step;
            new_shape.push_back(len);
            new_stride.push_back(orig_stride[i] * r.step);
            new_offset += start * orig_stride[i];
        }

        return tensrLens::lens<T>(lensW.data().lock(), new_shape, new_stride, new_offset);
    }
}