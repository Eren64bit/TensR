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
//---------------------------------Constructors
template<typename T>
tensr::Tensr<T>::Tensr(std::vector<size_t>& shape) : shape_(std::move(shape)) {
    stride_ = compute_strides(shape);
    total_size_ = compute_total_size(shape);
    rank_ = compute_rank(shape);

    data_ptr_ = std::make_shared<std::vector<T>>(total_size_);
    std::fill(data_ptr_->begin(), data_ptr_->end(), 0);
}

template<typename T>
tensr::Tensr<T>::Tensr(const std::vector<size_t>& shape, const std::vector<T>& data) : shape_(std::move(shape)) {
	stride_ = compute_strides(shape);
    total_size_ = compute_total_size(shape);
    rank_ = compute_rank(shape);

    data_ptr_ = std::make_shared<std::vector<T>>(std::move(data));

    if (data_ptr_->size() != total_size_) {
        throw std::runtime_error("Data and Total size does not match up");
    }
    
}
