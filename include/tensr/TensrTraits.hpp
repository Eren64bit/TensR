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
};

template <typename T>
struct TensrTraits<TensrLens<T>> {
    static bool is_contiguous(const Tensr<T>& t) {
        return false;
    }

    static void reshape(TensrLens<T>&, const std::vector<size_t>&) {
        throw std::runtime_error("reshape not supported on TensrLens");
    }
};