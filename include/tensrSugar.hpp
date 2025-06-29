#pragma once 

#include "tensrLazy.hpp"
#include "tensrLens.hpp"
#include "tensr.hpp"


#define DEFINE_TENSOR_BINARY_OP(op) \
template<typename T> \
tensr::Tensr<T> operator op (const tensr::Tensr<T>& a, const tensr::Tensr<T>& b) { \
    return tensrLazy::materialize(*(tensrLazy::leaf(a) op tensrLazy::leaf(b))); \
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