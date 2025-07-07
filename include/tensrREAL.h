#pragma once

#include "tensrBASE.h"

template<typename T>
class tensrREAL : public tensrBASE<tensrREAL, T> {
    public:
        // Constructor
        tensrREAL() = default;
        // Variadic constructor with data pointer
        // example: tensrREAL<float> tensor(3, 4, 5);
        template<typename... Args,
                typename = std::enable_if_t<(sizeof...(Args) > 1)>>
        tensrREAL(Args&&... args) 
            : tensrBASE<tensrREAL, T>() {
                static_assert((std::is_convertible_v<Args, size_t> && ...), "All dimensions must be size_t");
                this->shape_ = {static_cast<size_t>(args)...};
                this->strides_ = this->compute_strides(this->shape_);
                this->data_ = std::make_shared<std::vector<T>>(this->size());
                this->offset_ = 0; // Default offset is 0
                this->policy_ = tensrPolicy::TENSOR; // Default policy is TENSOR 
            }
        // Vector-based constructor
        // example: tensrREAL<float> tensor({3, 4, 5});
        // example with data: tensrREAL<float> tensor({3, 4, 5}, std::make_shared<std::vector<float>>(data_vector));
        tensrREAL(const std::vector<size_t>& shape, const std::shared_ptr<std::vector<T>>& data = nullptr)
            : tensrBASE<tensrREAL, T>() {
            this->shape_ = shape;
            this->strides_ = this->compute_strides(shape);
            this->data_ = data ? data : std::make_shared<std::vector<T>>(size());
            this->offset_ = 0; // Default offset is 0
            this->policy_ = tensrPolicy::TENSOR; // Default policy is TENSOR
        }

        // Override data() method to return the data pointer
        std::shared_ptr<std::vector<T>> data() const override {
            return this->data_; // Return the shared pointer to the data vector
        }
};