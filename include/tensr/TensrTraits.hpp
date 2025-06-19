#pragma once

#include "Tensr.hpp"
#include "Lens.hpp"
#include "util/TensrUtils.hpp"

template <typename T>
struct TensrTraits;


template <typename T>
struct TensrTraits<Tensr<T>> {
    static bool is_contiguous(const Tensr<T>& t) {
        return t.stride() == compute_strides(t.shape());
    }

    static void reshape(Tensr<T>& t, const std::vector<size_t>& shape) {
        t.set_shape(shape);
        t.set_stride(compute_strides(shape));
    }
    static void squeeze(Tensr<T>& t) {
        std::vector<size_t> new_shape;
        for (size_t i : t.shape()) {
            if (i == 1) {
                continue;
            } else {
                new_shape.push_back(i);
            }
        }
        t.set_shape(new_shape);
        t.set_stride(compute_strides(new_shape));
    }

    static void squeeze(Tensr<T>& t, int axis) {
        std::vector<size_t> new_shape;
        for (size_t i : t.shape()) {
            if (i == axis && t.shape()[i] == 1) {
                continue;
            } else {
                new_shape.push_back(i);
            }
        }
        t.set_shape(new_shape);
        t.set_stride(compute_strides(new_shape));
    }
};

template <typename T>
struct TensrTraits<TensrLens<T>> {
    static bool is_contiguous(const Tensr<T>& t) {
        return false;
    }

    static void reshape(TensrLens<T>&, const std::vector<size_t>&) {
        throw std::runtime_error("reshape not supported on TensrLens");
    }

    static void squeeze(TensrLens<T>& t) {
        std::vector<size_t> new_shape;
        for (size_t i : t.shape()) {
            if (i == 1) {
                continue;
            } else {
                new_shape.push_back(i);
            }
        }
        t.set_shape(new_shape);
        t.set_stride(compute_strides(new_shape));
    }

    static void squeeze(TensrLens<T>& t, int axis) {
        std::vector<size_t> new_shape;
        for (size_t i : t.shape()) {
            if (i == axis) {
                continue;
            } else {
                new_shape.push_back(i);
            }
        }
        t.set_shape(new_shape);
        t.set_stride(compute_strides(new_shape));
    }
};