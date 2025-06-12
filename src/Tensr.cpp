#include "../include/Tensr.h"

//API Functions//////////////////////////////
template<typename T, DeviceType Device>
Tensr<T, Device>::Tensr(std::vector<size_t> shape) {
    this->shape_ = shape;
    if (this->total_size_ == 0) throw std::runtime_error("Tensor size must be bigger than 0\n");
    //fill with zeros
    data_.resize(total_size_);
    std::fill(data_.begin(), data_.end(), 0);
}


//Helper Functions////////////////////////////
template<typename T, DeviceType Device>
void Tensr<T, Device>::compute_total_size_() {
    int result = 1;
    for (int i = 0; i < this->shape_.size(); i++) {
        result *= this->shape_[i];
    }
    this->size(result);
}

template<typename T, DeviceType Device>
void Tensr<T, Device>::compute_strides_() {
    for (int i = 0; i < this->stride_.size(); i++) {

    }
}

template<typename T, DeviceType Device>
void Tensr<T, Device>::compute_rank_() {
}

