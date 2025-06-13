#include "../include/Tensr.h"

//API Functions//////////////////////////////

//***************************************************************Constructers
template<typename T, DeviceType Device>
Tensr<T, Device>::Tensr(std::vector<size_t> shape) 
    :  shape_(std::move(shape)) {
    compute_total_size_();
    compute_rank_();
    compute_strides_();

    if (this->total_size_ == 0) throw std::runtime_error("error:Tensor size must be bigger than 0\n");
    //fill with zeros
    data_.resize(total_size_);
    std::fill(data_.begin(), data_.end(), 0);
    }

template<typename T, DeviceType Device>
Tensr<T, Device>::Tensr(std::vector<size_t> shape, std::vector<value_type> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
        compute_total_size_();
        compute_rank_();
        compute_strides_();
    }
//***************************************************************END

//***************************************************************at functions
template<typename T, DeviceType Device>
T& Tensr<T, Device>::at(const std::vector<size_t>& indices) {
    size_t flat_idx = this->flat_index_(indices);
    return data_[flat_idx];
}
template<typename T, DeviceType Device>
const T& Tensr<T, Device>::at(const std::vector<size_t>& indices) const {
    size_t flat_idx = this->flat_index_(indices);
    return data_[flat_idx];
}
//***************************************************************END

//***************************************************************Reshape function
template<typename T, DeviceType Device>
void Tensr<T, Device>::reshape(std::vector<size_t> new_shape) {
    size_t result = 1;
    for (int i = 0; i < new_shape.size(); i++) {
        result *= new_shape[i];
    }
    if (result != this->total_size_) {
        throw std::runtime_error("error:Shape size does not match");
    }
    this->shape_ = new_shape;
    this->compute_rank_();
    this->compute_strides_();
}
//***************************************************************END

//Helper Functions////////////////////////////
template<typename T, DeviceType Device>
size_t Tensr<T, Device>::flat_index_(const std::vector<size_t>& indices) {
    if (indices.size() != rank_) {
        throw std::runtime_error("error:Index dimensionality does not match tensor rank");
    }
    for (size_t i = 0; i < rank_; ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("error:Index out of bounds for dimension " + std::to_string(i));
        }
    }
    if (stride_.empty()) {
        throw std::runtime_error("error:Strides not initialized");
    }
    size_t result = 0;
    for (size_t i = 0; i < rank_ ; i++) {
        result += indices[i] * stride_[i];
    }
    return result;
}

template<typename T, DeviceType Device>
void Tensr<T, Device>::compute_total_size_() {
    if (shape_.empty()) {
        throw std::runtime_error("error:Shape Cannot be empty");
    }
    size_t result = 1;
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




//Carpe diem.