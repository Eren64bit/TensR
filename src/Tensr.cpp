#include "../include/Tensr.h"

//API Functions//////////////////////////////
template<typename T, DeviceType Device>
Tensr<T, Device>::Tensr(std::vector<size_t> shape) {

    this->shape_ = shape;
    this->compute_total_size_();
    this->compute_rank_();
    this->compute_strides_();

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
    this->total_size_ = result;
}

template<typename T, DeviceType Device>
void Tensr<T, Device>::compute_strides_() {
    this->stride_.clear();
    this->stride_.resize(this->rank_);
    this->stride_[rank_ - 1] = 1;

    for (int i = rank_ - 2; i >= 0; --i) {
        stride_[i] = stride_[i + 1] * shape_[i + 1];
    }
}

template<typename T, DeviceType Device>
void Tensr<T, Device>::compute_rank_() {
    this->rank_ = this->shape_.size();
}

