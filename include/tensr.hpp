#pragma once

#include <memory>
#include <vector>

namespace tensr {

template<typename T>
class TensrBase {
public:
    virtual const std::weak_ptr<std::vector<T>> data() const = 0; // return weak pointer to tensor data

    virtual const std::vector<size_t> shape() const = 0; // return tensor or lens shape
    virtual const std::vector<size_t> stride() const = 0; // return tensor or lens stride

    virtual const size_t size() const = 0; // return tensor or lens total size

    virtual const size_t rank() const = 0; // return tensor or lens rank(Shape size)
    virtual const size_t offset() const = 0; // return tensor or lens offset(distance from 0 index data)

};

template<typename T>
class Tensr : public TensrBase {
private:
    std::shared_ptr<std::vector<T>> data_ptr_; // data pointer to real data 

    std::vector<size_t> shape_; //Tensor shape
    std::vector<size_t> stride_; //tensor stride

    size_t total_size_; // Tensor total size
    size_t rank_; // Tensor rank (shape size)
    size_t offset_; // Tensor offset

    static_assert(std::is_arithmetic_v<T>, "error:Tensor type must be arithmetic"); // make sure T is arithmetic
    
public:

    explicit Tensr();
    Tensr();

    const std::weak_ptr<std::vector<size_t>> data() const override { return data_ptr_; } // return data_ptr make sure before use it, Dereference it

    const std::vector<size_t> shape() const override { return shape_; } // return size_t vector shape
    const std::vector<size_t> stride() const override { return stride_; } // return size_t vector stride

    const size_t size() const override { return total_size_; } // return size_t total size

    const size_t rank() const override { return rank_; } // return size_t rank (shape size)
    const size_t offset() const override { return offset_; } // return size_t offset 

    //------------------------------------------------------At functions
    T& at();
    const T& at();
    //------------------------------------------------------Setter functions
    void set_data();
    void set_shape();
    void set_stride();
    void set_total_size();
    void set_rank();
    void set_offset();
};

}