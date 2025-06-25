#pragma once

#include <vector>
#include <cstddef>

namespace tensr { template<typename T> class Tensr; }
namespace tensrLens { template<typename T> class lens; }

namespace tensrOps {
    //-------------------------------Transpose
    template<typename T>
    tensrLens::lens<T> transpose(const tensr::Tensr<T>&, const std::vector<size_t>&);

    template<typename T>
    tensrLens::lens<T> transpose(const tensr::Tensr<T>&);

    template<typename T>
    tensrLens::lens<T> transpose(const tensrLens::lens<T>&, const std::vector<size_t>&);

    template<typename T>
    tensrLens::lens<T> transpose(const tensrLens::lens<T>&);

    //-------------------------------Slice
    struct SliceRange {
        size_t start;
        size_t stop;
        size_t step;

        SliceRange() = default;
        SliceRange(size_t s, size_t e, size_t st = 1)
            : start(s), stop(e), step(st) {}
    };

    template<typename T>
    tensrLens::lens<T> slice(const tensr::Tensr<T>&, std::vector<SliceRange>&);

    template<typename T>
    tensrLens::lens<T> slice(const tensrLens::lens<T>& lensView, const std::vector<SliceRange>& ranges);
}