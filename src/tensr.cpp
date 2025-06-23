#pragma once

#include "../include/tensr.hpp"
#include "../include/TensrUtils.hpp"

//--------------------------------Set shape
template<typename T>
void tensr::Tensr<T>::set_shape(const std::vector<size_t>& tshape) {
    shape_ = std::move(tshape);
    stride_ = compute_strides(tshape);
    total_size_ = compute_total_size(tshape);
    rank_ = compute_rank(tshape);
}
