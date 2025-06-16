#pragma once
#include "Tensr.h"
#include <memory>


template<typename T, DeviceType Device>
class TensrLens {
private:
    std::weak_ptr<T> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;

    size_t total_size_;
    size_t offset_ = 0;
    size_t rank_;

    static_assert(std::is_arithmetic_v<T>, "error:Tensor type must be arithmetic");

public:

    TensrLens(Tensr<T, Device>& original, std::vector<size_t> view_shape)
        : data_(original), shape_(view_shape) {}

};