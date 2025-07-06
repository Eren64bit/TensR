#pragma once

#include "tensrBASE.h"

template<typename T>
class tensrREAL : public tensrBASE<tensrREAL, T> {
    public:
        // Constructor
        tensrREAL() = default;
        tensrREAL(const std::vector<size_t>& shape, const std::shared_ptr<std::vector<T>>& data = nullptr)
            : tensrBASE<tensrREAL, T>() {
            this->shape_ = shape;
            this->strides_ = compute_strides(shape);
            this->data_ = data ? data : std::make_shared<std::vector<T>>(size());
            this->offset_ = 0; // Default offset is 0
            this->policy_ = tensrPolicy::TENSOR; // Default policy is TENSOR
        }

        // Override data() method to return the data pointer
        std::shared_ptr<std::vector<T>> data() const override {
            return this->data_;
        }
};