#pragma once 

#include "tensrLazy.hpp"
#include "tensrLens.hpp"
#include "tensr.hpp"


//rework tensrSugar

#define DEFINE_TENSOR_BINARY_OP(op) \
template<typename T> \
tensr::Tensr<T> operator op (const tensr::Tensr<T>& a, const tensr::Tensr<T>& b) { \
    switch (tensr::get_mode()) \
    { \
    case tensr::Mode::NORMAL: \
        return tensrLazy::materialize(*(tensrLazy::leaf(a) op tensrLazy::leaf(b))); \
        break; \
    case tensr::Mode::LAZY: \
        \
        break; \
    case tensr::Mode::CUDA: \
        \
        break; \
    default: \
        break; \
    } \ 
} \
template<typename T> \
tensr::Tensr<T> operator op (const tensr::Tensr<T>& a, const tensrLens::lens<T>& b) { \
    return tensrLazy::materialize(*(tensrLazy::leaf(a) op tensrLazy::leaf(b))); \
} \
template<typename T> \
tensr::Tensr<T> operator op (const tensrLens::lens<T>& a, const tensr::Tensr<T>& b) { \
    return tensrLazy::materialize(*(tensrLazy::leaf(a) op tensrLazy::leaf(b))); \
} \
template<typename T> \
tensr::Tensr<T> operator op (const tensrLens::lens<T>& a, const tensrLens::lens<T>& b) { \
    return tensrLazy::materialize(*(tensrLazy::leaf(a) op tensrLazy::leaf(b))); \
}


DEFINE_TENSOR_BINARY_OP(+)
DEFINE_TENSOR_BINARY_OP(-)
DEFINE_TENSOR_BINARY_OP(*)
DEFINE_TENSOR_BINARY_OP(/)


#undef DEFINE_TENSOR_BINARY_OP