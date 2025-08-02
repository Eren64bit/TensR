#pragma once

#include <memory>
#include <stdexcept>
#include <vector>


template<typename T>
class tensrSTATIC {
protected:
    std::shared_ptr<std::vector<T>> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    size_t offset_;

public:
    
    virtual std::shared_ptr<std::vector<T>> data() const = 0;

    const std::vector<size_t>& shape() { return shape_; }
    const std::vector<size_t>& strides() { return strides_; }
    
    size_t offset() const { return offset_; }

};