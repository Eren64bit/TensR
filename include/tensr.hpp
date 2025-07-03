#pragma once


#include "tensrBase.hpp"
#include "tensrUtils.hpp"
#include "indexUtils.hpp"
#include "tensrLens.hpp"
#include "tensrOps_decl.hpp"


namespace tensr {

    enum class Mode { NORMAL, LAZY, CUDA};

    static Mode currentMode = Mode::NORMAL;

    void set_mode(Mode m) {
        currentMode = m;
    }

    Mode get_mode() {
        return currentMode;
    }

template<typename T>
class Tensr : public TensrBase<T> {
private:
    std::shared_ptr<std::vector<T>> data_ptr_; // data pointer to real data 

    std::vector<size_t> shape_; //Tensor shape
    std::vector<size_t> stride_; //tensor stride

    size_t total_size_; // Tensor total size
    size_t rank_; // Tensor rank (shape size)
    size_t offset_; // Tensor offset

    static_assert(std::is_arithmetic_v<T>, "error:Tensor type must be arithmetic"); // make sure T is arithmetic
    
public:

    explicit Tensr(const std::vector<size_t>& shape) : shape_(std::move(shape)) { // Tensr Constructer
        stride_ = compute_strides(shape);
        total_size_ = compute_total_size(shape);
        rank_ = compute_rank(shape);
        offset_ = 0;

        data_ptr_ = std::make_shared<std::vector<T>>(total_size_);
        std::fill(data_ptr_->begin(), data_ptr_->end(), 0);
    }

    explicit Tensr(const std::vector<size_t>& shape, const std::vector<T>& data) : shape_(std::move(shape)) { // Tensr Constructer with data
	    stride_ = compute_strides(shape);
        total_size_ = compute_total_size(shape);
        rank_ = compute_rank(shape);
        offset_ = 0;

        data_ptr_ = std::make_shared<std::vector<T>>(std::move(data));

        if (data_ptr_->size() != total_size_) {
            throw std::runtime_error("Data and Total size does not match up");
        }
    }

    explicit Tensr(const std::vector<size_t>& shape, const T& fill_value) : shape_(std::move(shape)) {
        stride_ = compute_strides(shape);
        total_size_ = compute_total_size(shape);
        rank_ = compute_rank(shape);
        offset_ = 0;

        data_ptr_ = std::make_shared<std::vector<T>>(total_size_);
        fill(fill_value);
    }

    std::weak_ptr<std::vector<T>> data() const override { return data_ptr_; } // return data_ptr make sure before use it, Dereference it

    const std::vector<size_t>& shape() const override { return shape_; } // return size_t vector shape
    const std::vector<size_t>& stride() const override { return stride_; } // return size_t vector stride

    size_t size() const override { return total_size_; } // return size_t total size

    size_t rank() const override { return rank_; } // return size_t rank (shape size)
    size_t offset() const override { return offset_; } // return size_t offset 

    //------------------------------------------------------At functions
    T& at(const std::vector<size_t>& indices) override {
        size_t flat_index = indexUtils::flat_index(indices, shape_, stride_);
        return (*data_ptr_)[flat_index];
    }
    const T& at(const std::vector<size_t>& indices) const override {
        size_t flat_index = indexUtils::flat_index(indices, shape_, stride_);
        return (*data_ptr_)[flat_index];
    }
    //------------------------------------------------------At linear functions
    T& at_linear(size_t linear_idx) {
        if (!is_contiguous()) {
            throw std::runtime_error("at_linear() called on non-contiguous tensor");
        }
        return (*data_ptr_)[linear_idx];
    }

    const T& at_linear(size_t linear_idx) const {
        if (!is_contiguous()) {
            throw std::runtime_error("at_linear() called on non-contiguous tensor");
        }
        return (*data_ptr_)[linear_idx];
    }
    //------------------------------------------------------Setter functions
    void set_shape(const std::vector<size_t>& tshape) {
        if (compute_total_size(tshape) != compute_total_size(shape_)) throw std::invalid_argument("Cannot reshape tensor: total size mismatch.");
        shape_ = std::move(tshape);
        stride_ = compute_strides(tshape);
        total_size_ = compute_total_size(tshape);
        rank_ = compute_rank(tshape);
    }

    void set_stride(const std::vector<size_t>& tstride) { stride_ = tstride; }
    void set_total_size(const size_t& tsize) { total_size_ = tsize; }
    void set_rank(const size_t& trank) { rank_ = trank; }
    void set_offset(const size_t& toffset) { offset_ = toffset; }

    //-----------------------------------------------------Bool functions
    bool is_contiguous() const {
        auto expected = compute_strides(shape_);
        return stride_ == expected;
    }
    
    //-------------------------------------Free functions implementations
    tensrLens::lens<T> reshape(const std::vector<size_t>& target_shape) {
        return tensrOps::reshape(*this, target_shape);
    }

    tensrLens::lens<T> transpose(const std::vector<size_t> perm) {
        return tensrOps::transpose(*this, perm);
    }

    tensrLens::lens<T> slice(const std::vector<tensrOps::SliceRange>& ranges) {
        return tensrOps::slice(*this, ranges);
    }

    tensrLens::lens<T> squeeze() {
        return tensrOps::squeeze(*this);
    }

    tensrLens::lens<T> unsqueeze(const int axis) {
        return tensrOps::unsqueeze(*this, axis);
    }

    //-------------------------------------view()
    tensrLens::lens<T> view() const {
        return tensrLens::lens<T>(data_ptr_, shape_, stride_, offset_);
    }
    //-------------------------------------Fill()
    void fill(const T& value) {
        if (is_contiguous()) {
            for (size_t i = 0; i < total_size_; ++i)
                at_linear(i) = value;
        } else {
            for (size_t i = 0; i < size(); ++i) {
                auto idx = indexUtils::unflatten_index(i, shape_);
                at(idx) = value;
            }
        }
    }

    //-------------------------------------info()
    void info() const {
        std::cout << "Tensor Info:\n";
        std::cout << "  Shape: ";
        for (auto s : shape_) std::cout << s << " ";
        std::cout << "\n  Stride: ";
        for (auto s : stride_) std::cout << s << " ";
        std::cout << "\n  Offset: " << offset_;
        std::cout << "\n  Size: " << total_size_;
        std::cout << "\n  Contiguous: " << (is_contiguous() ? "Yes" : "No") << std::endl;
    }
};

}

