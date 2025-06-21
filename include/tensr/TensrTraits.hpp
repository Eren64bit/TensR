#pragma once

#include "Tensr.hpp"
#include "Lens.hpp"
#include "util/TensrUtils.hpp"
#include "ops/Broadcast.hpp"
#include <algorithm>

template <typename T>
struct TensrTraits;


template <typename T>
struct TensrTraits<Tensr<T>> {
    static bool is_squeezable(Tensr& t) {
        return std::any_of(t.shape().start(), t.shape().end(), [](size_t dim) { return dim == 1; });
    }

    static bool is_contiguous(const Tensr<T>& t) {
        return t.stride() == compute_strides(t.shape());
    }
    //--------------------------------Broadcast_to
    static TensrLens<T> broadcas_to(const Tensr<T>& t, const std::vector<size_t> target_shape) { // fix
        std::vector<size_t> common_shape = compute_broadcast_shape(t.shape(), target_shape);

        TensrLens<T> lens;
        
        

    }

    //--------------------------------Reshape
    static void reshape(Tensr<T>& t, const std::vector<size_t>& shape) {
        t.set_shape(shape);
        t.set_stride(compute_strides(shape));
    }

    
    //--------------------------------Squeeze
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
        if (axis < 0 || axis > t.shape().size()) throw std::runtime_error("Axis out of range");
        if (t.shape()[axis] != 1) throw std::runtime_error("Cannot squeeze axis with size != 1");
        for (int i = 0; i < t.shape().size(); i++) {
            if (t.shape()[i] == 1 && i == axis) continue;
            new_shape.push_back(t.shape()[i]);
        }
        t.set_shape(new_shape);
        t.set_stride(compute_strides(new_shape));
    }

    //--------------------------------Unsqueeze
    static void unsqueeze(Tensr<T>& t, int axis) {
        if (axis < 0 || axis > t.shape().size()) {
            throw std::runtime_error("Axis out of range");
        }
        std::vector<size_t> new_shape;
        for (size_t i = 0; i <= t.shape().size(); ++i) {
            if (i == axis) {
                new_shape.push_back(1);
            }
            if (i < t.shape().size()) {
                new_shape.push_back(t.shape()[i]);
            }
        }
        t.set_shape(new_shape);
        t.set_stride(compute_strides(new_shape));
    }


};

template <typename T>
struct TensrTraits<TensrLens<T>> {
    static bool is_squeezable(TensrLens& t) {
        return std::any_of(t.shape().begin(), t.shape().end(), [](size_t dim) { return dim == 1; });
    }

    static bool is_contiguous(const TensrLens<T>& t) {
        return false;
    }

    static void reshape(TensrLens<T>&, const std::vector<size_t>&) {
        throw std::runtime_error("reshape not supported on TensrLens");
    }

    static void squeeze(TensrLens<T>& t, int) {
        throw std::runtime_error("squeeze not supported on TensrLens");
    }

    static void unsqueeze(TensrLens<T>&, int) {
        throw std::runtime_error("Unsqueeze not supported on TensrLens");
    }

};