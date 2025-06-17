#pragma once

#include "vector"
#include <memory>

template<typename T>
class TensrBase {
public:
    virtual const std::weak_ptr<std::vector<T>> data() const = 0;
    virtual std::weak_ptr<std::vector<T>> mutable_data() = 0;

    virtual const std::vector<size_t>& shape() const = 0;
    virtual size_t size() const = 0;

    virtual std::vector<size_t> unflatten_index_(size_t index) const = 0; 
};

template<typename T>
class Tensr : public TensrBase<T> { 
private:
    std::vector<T> data_;

    std::vector<size_t> shape_;
    std::vector<size_t> stride_;

    size_t total_size_;
    size_t rank_;
    size_t offset_;

    static_assert(std::is_arithmetic_v<T>, "error:Tensor type must be arithmetic");

public:
    using value_type = T;

    explicit Tensr(std::vector<size_t> shape);
    Tensr(std::vector<size_t> shape, std::vector<value_type> data);

    const std::vector<size_t>& shape() const override;
    const std::weak_ptr<T> data() override; 
    size_t size() const override { return total_size_; }

};