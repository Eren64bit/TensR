#pragma once

#include <memory>
#include <vector>
#include <cstddef>

template<typename T>
class TensrBase {
public:
    virtual std::weak_ptr<std::vector<T>> data() const = 0; // return weak pointer to tensor data

    virtual const std::vector<size_t> shape() const = 0; // return tensor or lens shape
    virtual const std::vector<size_t> stride() const = 0; // return tensor or lens stride

    virtual size_t size() const = 0; // return tensor or lens total size

    virtual size_t rank() const = 0; // return tensor or lens rank(Shape size)
    virtual size_t offset() const = 0; // return tensor or lens offset(distance from 0 index data)

    virtual T& at(const std::vector<size_t>& indices) = 0;
    virtual const T& at(const std::vector<size_t>& indices) const = 0;

    virtual ~TensrBase() = default; // virtual destructor for base class
};