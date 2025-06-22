#pragma once

#include "Tensr.hpp"

template<typename T>
class TensrLens : public TensrBase<T> {
private:
    std::weak_ptr<std::vector<T>> data_;

    std::vector<size_t> shape_;
    std::vector<size_t> stride_;

    size_t total_size_;
    size_t rank_;
    size_t offset_;

public:
    using value_type = T;

    TensrLens(std::shared_ptr<std::vector<T>> data_ptr, std::vector<size_t> shape, std::vector<size_t> stride, size_t offset);

    const std::weak_ptr<std::vector<T>> data() const override {
        return data_;
    }

    std::weak_ptr<std::vector<T>> mutable_data() override {
        return data_.lock();
    }
    const std::vector<size_t>& shape() const override { return shape_; }
    const std::vector<size_t>& stride() const override { return stride_; }

    value_type& at(const std::vector<size_t>& indices);
    const value_type& at(const std::vector<size_t>& indices) const;

    value_type& at_flat(size_t flat_index) {
        auto shared_data = data_.lock();
        return (*shared_data)[offset_ + flat_index];
    }

    const value_type& at_flat(size_t flat_index) const {
        auto shared_data = data_.lock();
        return (*shared_data)[offset_ + flat_index];
    }

    void set_data(const std::shared_ptr<std::vector<size_t>> data_ptr) { this->data_ = std::move(data_ptr); }
    void set_shape(const std::vector<size_t> shape) { this->shape_ = shape; }
    void set_stride(const std::vector<size_t> stride) { this->stride_ = stride; }
    void set_offset(const size_t offset) { this->offset_ = offset; }
    
};