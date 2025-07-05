#pragma once 

#include "tensrLazy.hpp"
#include "tensrLens.hpp"
#include "tensr.hpp"
#include "CUDA/tensrCUDA.h"

tensr::Mode resolve_mode(tensr::Mode lmd, tensr::Mode rmd) {
    if (lmd == tensr::Mode::CUDA || rmd == tensr::Mode::CUDA) {
        return tensr::Mode::CUDA;
    } else if (lmd == tensr::Mode::LAZY || rmd == tensr::Mode::LAZY) {
        return tensr::Mode::LAZY;
    } else {
        return tensr::Mode::NORMAL;
    }
}


#define DEFINE_BINARY_OP(opname, opfunc) \
template<typename T> \
auto operator opname (const tensr::Tensr<T>& lhs, const tensr::Tensr<T>& rhs) { \
    auto mode = resolve_mode(lhs.mode(), rhs.mode()); \
    auto expr = tensrLazy::leaf(lhs) opname tensrLazy::leaf(rhs);   \
        \
    if (mode == tensr::Mode::NORMAL) {  \
        return tensrLazy::materialize(*expr, mode); \
    } else if (mode == tensr::Mode::LAZY){  \
        return expr;    \
    } else if (mode == tensr::Mode::CUDA) { \
        return tensrCUDA::tensrCUDA<T>::compute(opname, lhs, rhs);  \
    }   \
}   \


DEFINE_BINARY_OP(+, add)
DEFINE_BINARY_OP(-, subtract)
DEFINE_BINARY_OP(*, multiply)
DEFINE_BINARY_OP(/, a / b)

#undef DEFINE_BINARY_OP