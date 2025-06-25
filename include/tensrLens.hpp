#pragma once


#include "tensrBase.hpp"
#include "tensrUtils.hpp"
#include "indexUtils.hpp"
#include "tensrOps.hpp"

namespace tensrLens{

template<typename T>
class lens : public TensrBase<T> {
private:
    std::weak_ptr<std::vector<T>> data_ptr_; // pointer to real data
    std::shared_ptr<std::vector<T>> cached_ptr_; // cached pointer for big tensors

    std::vector<size_t> shape_;
    std::vector<size_t> stride_;

    size_t offset_;
    size_t total_size_;
    size_t rank_;

    static_assert(std::is_arithmetic_v<T>, "error:Tensor type must be arithmetic"); // make sure T is arithmetic

public:

    lens(const std::shared_ptr<std::vector<T>>& data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, const size_t& offset) 
        : data_ptr_(data), shape_(shape), stride_(stride), offset_(offset) { // Consstructor 
            total_size_ = compute_total_size(shape_);
            rank_ = compute_rank(shape_);
    }
    std::weak_ptr<std::vector<T>> data() const override { return data_ptr_; } // return weak pointer make sure use lock() before accses data

    const std::vector<size_t> shape() const override { return shape_; }
    const std::vector<size_t> stride() const override { return stride_; }

    size_t size() const override { return total_size_; }
    
    size_t rank() const override { return rank_; }
    size_t offset() const override { return offset_; }

    T& at(const std::vector<size_t>& indices) override {
        size_t flat_index = indexUtils::flat_index(indices, shape_, stride_);

        size_t final_index = offset_ + flat_index;

        if (cached_ptr_) {
            return (*cached_ptr_)[final_index];
        } else {
            auto ps = data_ptr_.lock();
            if (!ps) throw std::runtime_error("Data expired");
            return (*ps)[final_index];
        }
    }
    const T& at(const std::vector<size_t>& indices) const override {
        size_t flat_index = indexUtils::flat_index(indices, shape_, stride_);

        size_t final_index = offset_ + flat_index;

        if (cached_ptr_) { //if cached this function will be much faster.
            return (*cached_ptr_)[final_index];
        } else {
            auto ps = data_ptr_.lock();
            if (!ps) throw std::runtime_error("Data expired");
            return (*ps)[final_index];
        }
    }
    //-------------------------------
    void cache_data_ptr() {
        auto sp = data_ptr_.lock();
        if (!sp) throw std::runtime_error("Data expired");
        cached_ptr_ = sp;
    }

    void clear_cache() {
        cached_ptr_.reset();
    }


    //-------------------------------Bool functions
    bool is_valid() const {
        return !data_ptr_.expired();
    }
    size_t use_count() const {
        return data_ptr_.use_count();
    }
    bool is_contiguous() const {
        auto expected = compute_strides(shape_);
        return stride_ == expected;
    }


    void info() const {
        std::cout << "Lens Info:\n";
        std::cout << "  Shape: ";
        for (auto s : shape_) std::cout << s << " ";
        std::cout << "\n  Stride: ";
        for (auto s : stride_) std::cout << s << " ";
        std::cout << "\n  Offset: " << offset_;
        std::cout << "\n  Size: " << total_size_;
        std::cout << "\n  Valid: " << (is_valid() ? "Yes" : "No");
        std::cout << "\n  Use count: " << use_count();
        std::cout << "\n  Contiguous: " << (is_contiguous() ? "Yes" : "No") << std::endl;
    }

    //-------------------------------------Free functions implementations
    tensrLens::lens<T> transpose(const std::vector<size_t> perm) {
        return tensrOps::transpose(*this, perm);
    }
};

}