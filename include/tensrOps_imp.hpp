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


            int start = indexUtils::normalize_slice_index(r.start, dim, true);
            int stop = indexUtils::normalize_slice_index(r.stop, dim, true);

            if (start < 0 || stop < 0 || start >= static_cast<int>(dim) || stop > static_cast<int>(dim) || r.step == 0) {
                throw std::out_of_range("Invalid (normalized) slice range");
            }
            size_t len;
            if ((r.step > 0 && start >= stop) || (r.step < 0 && start <= stop)) {
                len = 0;
            } else {
                len = (std::abs(stop - start) + std::abs(r.step) - 1) / std::abs(r.step);
            }
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


            int start = indexUtils::normalize_slice_index(r.start, dim, true);
            int stop = indexUtils::normalize_slice_index(r.stop, dim, true);

            if (start < 0 || stop < 0 || start >= static_cast<int>(dim) || stop > static_cast<int>(dim) || r.step == 0) {
                throw std::out_of_range("Invalid (normalized) slice range");
            }
            size_t len;
            if ((r.step > 0 && start >= stop) || (r.step < 0 && start <= stop)) {
                len = 0;
            } else {
                len = (std::abs(stop - start) + std::abs(r.step) - 1) / std::abs(r.step);
            }
            new_shape.push_back(len);
            new_stride.push_back(orig_stride[i] * r.step);
            new_offset += start * orig_stride[i];
        }

        return tensrLens::lens<T>(lensW.data().lock(), new_shape, new_stride, new_offset);
    }

    //------------------------------Reshape()
    template<typename T>
    tensrLens::lens<T> reshape(const tensr::Tensr<T>& tensor, const std::vector<size_t>& shape) {
        auto lens = tensor.view();
        lens.set_shape(shape);
        return lens;
    }

    template<typename T>
    tensrLens::lens<T> reshape(const tensrLens::lens<T>& lens, const std::vector<size_t>& shape) {
        auto copy = lens;
        copy.set_shape(shape);
        return copy;
    }

    //-----------------------------------Squeeze()
    template<typename T>
    tensrLens::lens<T> squeeze(const tensr::Tensr<T>& tensor) {
        std::vector<size_t> new_shape;
        for (size_t i : tensor.shape()) {
            if (i == 1) {
                continue;
            } else {
                new_shape.push_back(i);
            }
        }
        std::vector<size_t> new_stride = compute_strides(new_shape);
        return tensrLens::lens<T>(tensor.data().lock(), new_shape, new_stride, tensor.offset());
    }

    template<typename T>
    tensrLens::lens<T> squeeze(const tensrLens::lens<T>& lens) {
        std::vector<size_t> new_shape;
        for (size_t i : lens.shape()) {
            if (i == 1) {
                continue;
            } else {
                new_shape.push_back(i);
            }
        }
        std::vector<size_t> new_stride = compute_strides(new_shape);
        return tensrLens::lens<T>(lens.data().lock(), new_shape, new_stride, lens.offset());
    }

    template<typename T>
    tensrLens::lens<T> squeeze(const tensr::Tensr<T>& tensor, const int axis) {
        
        if (axis < 0 || axis > tensor.shape().size()) throw std::runtime_error("squeeze error: axis out of range");
        if (tensor.shape()[axis] != 1) throw std::runtime_error("squeeze error: Cannot squeeze axis with size != 1");

        std::vector<size_t> new_shape;
        for (size_t i = 0; i < tensor.shape().size(); i++) {
            if (i == axis && tensor.shape()[i] == 1) continue;
            new_shape.push_back(tensor.shape()[i]);
        }

        std::vector<size_t> new_stride = compute_strides(new_shape);
        return tensrLens::lens<T>(tensor.data().lock(), new_shape, new_stride, tensor.offset());
    }

    template<typename T>
    tensrLens::lens<T> squeeze(const tensrLens::lens<T>& lens, const int axis) {

        if (axis < 0 || axis > lens.shape().size()) throw std::runtime_error("squeeze error: axis out of range");
        if (lens.shape()[axis] != 1) throw std::runtime_error("squeeze error: Cannot squeeze axis with size != 1");

        std::vector<size_t> new_shape;
        for (size_t i = 0; i < lens.shape().size(); i++) {
            if (i == axis && lens.shape()[i] == 1) continue;
            new_shape.push_back(lens.shape()[i]);
        }

        std::vector<size_t> new_stride = compute_strides(new_shape);
        return tensrLens::lens<T>(lens.data().lock(), new_shape, new_stride, lens.offset());
    }
    

    //------------------------------Unsqueeze()
    template<typename T>
    tensrLens::lens<T> unsqueeze(const tensr::Tensr<T>& tensor, const int axis) {
        if (axis < 0 || axis > tensor.shape().size()) throw std::runtime_error("unsqueeze error: axis out of range");
        std::vector<size_t> new_shape;
        new_shape.reserve(tensor.shape().size() + 1);
        for (size_t i = 0; i < tensor.shape().size(); i++) {
            if (i == axis) new_shape.push_back(1);
            if (i < tensor.shape.size()) new_shape.push_back(tensor.shape()[i]);
        }

        std::vector<size_t> new_stride = compute_strides(new_shape);
        return tensrLens::lens<T>(tensor.data().lock(), new_shape, new_stride, tensor.offset());
    }

    template<typename T>
    tensrLens::lens<T> unsqueeze(const tensrLens::lens<T>& lens, const int axis) {
        if (axis < 0 || axis > lens.shape().size()) throw std::runtime_error("unsqueeze error: axis out of range");
        std::vector<size_t> new_shape;
        new_shape.reserve(lens.shape().size() + 1);
        for (size_t i = 0; i < lens.shape().size(); i++) {
            if (i == axis) new_shape.push_back(1);
            if (i < lens.shape.size()) new_shape.push_back(lens.shape()[i]);
        }

        std::vector<size_t> new_stride = compute_strides(new_shape);
        return tensrLens::lens<T>(lens.data().lock(), new_shape, new_stride, lens.offset());
    }

}