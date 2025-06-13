#pragma once

#include <vector>
#include <memory>
#include <stdexcept>

#include "DeviceType.h"


template<typename T, DeviceType Device = DeviceType::CPU>

class Tensr {
private:
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    std::vector<T> data_;

    size_t total_size_;
    size_t offset_ = 0;
    size_t rank_;

    static_assert(std::is_arithmetic_v<T>, "Tensor type must be arithmetic");

    void compute_total_size_();
    void compute_strides_();
    void compute_rank_();

    void flat_index_(std::vector<size_t> indices);
public:
    using value_type = T;

    explicit Tensr(std::vector<size_t> shape);
    Tensr(std::vector<size(t)> shape, std::vector<value_type> data);


    //Basic Functions
    size_t const size() const {return total_size_;}
    size_t const offset() const {return offset_;}
    size_t const rank() const {return rank_;}

    //
    value_type& operator[](size_t idx);
    const value_type& operator[](size_t idx) const;

    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<T>& data() const { return *data_; }

};

/*
shape = [2,3]

data_shape = {
    0,0,0
    0,0,0
}
stride
data_bus = 0,0,0,0,0,0


*/