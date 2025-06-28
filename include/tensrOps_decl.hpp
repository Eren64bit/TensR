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
        int start = 0;
        int stop = 0;
        int step = 1;

        SliceRange() = default;
        SliceRange(int s, int e, int st = 1)
            : start(s), stop(e), step(st) {}
    };
    
    template<typename T>
    tensrLens::lens<T> slice(const tensr::Tensr<T>& tensor, const std::vector<SliceRange>& ranges);

    template<typename T>
    tensrLens::lens<T> slice(const tensrLens::lens<T>& lensW, const std::vector<SliceRange>& ranges);

    //------------------------------Reshape()
    template<typename T>
    tensrLens::lens<T> reshape(const tensr::Tensr<T>& tensor, const std::vector<size_t>& shape);

    template<typename T>
    tensrLens::lens<T> reshape(const tensrLens::lens<T>& lens, const std::vector<size_t>& shape);

    //-------------------------------Squeeze()
    template<typename T>
    tensrLens::lens<T> squeeze(const tensr::Tensr<T>& tensor);

    template<typename T>
    tensrLens::lens<T> squeeze(const tensrLens::lens<T>& lens);

    template<typename T>
    tensrLens::lens<T> squeeze(const tensr::Tensr<T>& tensor, const int axis);

    template<typename T>
    tensrLens::lens<T> squeeze(const tensrLens::lens<T>& lens, const int axis);

    //-------------------------------Unsqueeze()
    template<typename T>
    tensrLens::lens<T> unsqueeze(const tensr::Tensr<T>& tensor, const int axis);

    template<typename T>
    tensrLens::lens<T> unsqueeze(const tensrLens::lens<T>& lens, const int axis);
}