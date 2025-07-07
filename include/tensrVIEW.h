#pragma once

#include "tensrBASE.h"

template<typename T>
class tensrVIEW : public tensrBASE<tensrVIEW, T> {
public:
    // Empty constructor
    tensrVIEW() = default;

    // Variadic constructor with data pointer
    // Example: tensrVIEW<float> view(3, 4, 5);
    // This constructor allows creating a view with specified dimensions
    template<typename... Args,
             typename = std::enable_if_t<(sizeof...(Args) > 1)>>
    tensrVIEW(const std::shared_ptr<std::vector<T>>& data, Args&&... dims)
        : tensrBASE<tensrVIEW, T>() {
        static_assert((std::is_convertible_v<Args, size_t> && ...), "All dimensions must be size_t");
        static_assert(sizeof...(dims) > 0, "Shape dimensions are required.");
        this->shape_ = {static_cast<size_t>(dims)...};
        this->strides_ = this->compute_strides(this->shape_);
        this->offset_ = 0;
        this->policy_ = tensrPolicy::VIEW;
        if (!data) throw std::invalid_argument("tensrVIEW requires non-null data pointer.");
        this->data_ = data; // Attach to external data
    }

    // Vector-based constructor
    // Example: tensrVIEW<float> view({3, 4, 5}, data_vector);
    // This constructor allows creating a view with specified shape and data pointer
    tensrVIEW(const std::vector<size_t>& shape,
              const std::shared_ptr<std::vector<T>>& data)
        : tensrBASE<tensrVIEW, T>() {
        this->shape_ = shape;
        this->strides_ = this->compute_strides(shape);
        this->data_ = data;
        this->offset_ = 0;
        this->policy_ = tensrPolicy::VIEW;
    }

    // Override data() method
    std::shared_ptr<std::vector<T>> data() const override {
        return this->data_;
    }
};
