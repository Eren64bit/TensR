#pragma once

#include "tensr_static.h"

template<typename T>
class tensr_view : public tensr_static<T> {
public:
    // Constructor that takes a shared pointer to data and shape
    tensr_view(std::shared_ptr<T[]> data, const std::vector<size_t>& shape, size_t offset = 0)
        : tensr_static<T>(shape, offset) {
        if (!data) throw std::invalid_argument("Data pointer must not be null.");
        this->data_ = data;
        this->strides_ = tensr_utils::compute_strides(shape);
    }

    // Override the data method to return the view's data
    [[nodiscard]] std::shared_ptr<T[]> data() const override {
        return this->data_;
    }

};