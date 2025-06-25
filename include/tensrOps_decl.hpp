#pragma once

#include <vector>

namespace tensr { template<typename T> class Tensr; }
namespace tensrLens { template<typename T> class lens; }

namespace tensrOps {
    template<typename T>
    tensrLens::lens<T> transpose(const tensr::Tensr<T>&, const std::vector<size_t>&);

    template<typename T>
    tensrLens::lens<T> transpose(const tensrLens::lens<T>&, const std::vector<size_t>&);
}