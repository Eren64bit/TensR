#pragma once

#include <memory>
#include <stdexcept>
#include "tensr_utils.h"


// TensorStatic<T> represents a fixed-shape, fixed-type tensor.
// It is an abstract base class that provides a common interface for tensors
// with static shapes and types. Derived classes must implement the data() method
template<typename T>
class tensr_static {
protected:
    std::shared_ptr<T[]> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    size_t offset_;

public:
    explicit tensr_static(const std::vector<size_t>& shape, size_t offset = 0)
        : shape_(shape), offset_(offset) {
        if (shape.empty()) throw std::invalid_argument("Shape must not be empty.");

        strides_ = tensr_utils::compute_strides(shape_);

        size_t total = tensr_utils::compute_size(shape_);
        data_ = std::shared_ptr<T[]>(new T[total]);
    }

    explicit tensr_static(const std::vector<size_t>& shape, const T data[], size_t offset = 0)
        : shape_(shape), offset_(offset) {
        if (shape.empty()) throw std::invalid_argument("Shape must not be empty.");

        strides_ = tensr_utils::compute_strides(shape_);

        size_t total = tensr_utils::compute_size(shape_);
        data_ = std::shared_ptr<T[]>(new T[total]);
        for (size_t i = 0; i < total; ++i) {
            data_[i] = data[i];
        }
    }
    
    [[nodiscard]] virtual std::shared_ptr<T[]> data() const {
        if (!data_) throw std::runtime_error("Data is not initialized.");
        return data_;
    }

    [[nodiscard]] T* raw_data() const {
        return data_.get();
    }

    [[nodiscard]] const std::vector<size_t>& shape() const { return shape_; }
    [[nodiscard]] const std::vector<size_t>& strides() const { return strides_; }
    
    size_t offset() const { return offset_; }

    // Access element at given indices
    T& at(const std::vector<size_t>& indices) {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Indices size must match shape size.");
        }
        size_t index = tensr_utils::flatten_index(shape_, strides_, indices);
        return data_[index + offset_];
    }

    const T& at(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Indices size must match shape size.");
        }
        size_t index = tensr_utils::flatten_index(shape_, strides_, indices);
        return data_[index + offset_];
    }

    T& operator()(const std::vector<size_t>& indices) {
        return at(indices);
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        return at(indices);
    }
    
    // Fill functions
    void fill(const T& value = T(0)) {
        fill_custom(value);
    }

    void fill_zeros() {
        if (!data_) throw std::runtime_error("Cannot fill: data is null.");
        tensr_utils::fill_data<T>::zeros(data_.get(), tensr_utils::compute_size(shape_));
    }
    void fill_custom(const T& value) {
        if (!data_) throw std::runtime_error("Cannot fill: data is null.");
        tensr_utils::fill_data<T>::custom(data_.get(), tensr_utils::compute_size(shape_), value);
    }

};