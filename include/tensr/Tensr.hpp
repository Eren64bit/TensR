#pragma once

#include <vector>
#include <memory>

template<typename T>
class TensrBase {
public:
    virtual const std::weak_ptr<std::vector<T>> data() const = 0;
    virtual std::weak_ptr<std::vector<T>> mutable_data() = 0;

    virtual const std::vector<size_t>& shape() const = 0;
    virtual const std::vector<size_t>& stride() const = 0;

    virtual const T& at(const std::vector<size_t>& idx) const = 0;

};

template<typename T>
class Tensr : public TensrBase<T> { 
private:
    std::shared_ptr<std::vector<T>> data_ptr_;

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

    const std::vector<size_t>& shape() const override { return shape_; }
    const std::vector<size_t>& stride() const override { return stride_; }
    const std::weak_ptr<std::vector<T>> data() const override { return data_ptr_; }
    std::weak_ptr<std::vector<T>> mutable_data() override { return data_ptr_; }
    size_t size() const override { return total_size_; }

    //at()
    value_type& at(const std::vector<size_t>& indices);
    const value_type& at(const std::vector<size_t>& indices) const override;


    //setter functions
    void set_shape(const std::vector<size_t> shape) { this->shape_ = shape; }
    void set_stride(const std::vector<size_t> stride) { this->stride_ = stride; }
};