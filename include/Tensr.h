#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <type_traits>

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

    static_assert(std::is_arithmetic_v<T>, "error:Tensor type must be arithmetic");

    void compute_total_size_();
    void compute_strides_();
    void compute_rank_();

    size_t flat_index_(const std::vector<size_t>& indices) const;
    std::vector<size_t> unflaten_index_(const size_t idx);
public:
    using value_type = T;

    explicit Tensr(std::vector<size_t> shape);
    Tensr(std::vector<size_t> shape, std::vector<value_type> data);


    //Basic Functions
    size_t  size() const {return total_size_;}
    size_t  offset() const {return offset_;}
    size_t  rank() const {return rank_;}

    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& stride() const { return stride_; }

    const std::vector<value_type>& data() const { return data_; }
    std::vector<value_type>& mutable_data() { return data_; }

    value_type& at(const std::vector<size_t>& indices);
    const value_type& at(const std::vector<size_t>& indices) const;

    value_type& operator[](const std::vector<size_t>& idx) { return at(idx); }
    const value_type& operator[](const std::vector<size_t>& idx) const { return at(idx); }

    void reshape(std::vector<size_t> new_shape);

};

