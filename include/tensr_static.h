#pragma once

#include <memory>
#include <stdexcept>


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
    tensr_static(const std::vector<size_t>& shape, const T data[], size_t offset = 0)
        : shape_(shape), offset_(offset) {
        if (shape.empty()) throw std::invalid_argument("Shape must not be empty.");
        
    }
    
    virtual std::shared_ptr<T[]> data() const = 0;

    const std::vector<size_t>& shape() { return shape_; }
    const std::vector<size_t>& strides() { return strides_; }
    
    size_t offset() const { return offset_; }

};