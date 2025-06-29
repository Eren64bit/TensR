#pragma once


#include "tensrBase.hpp"
#include "tensrUtils.hpp"
#include "indexUtils.hpp"
#include "tensrOps_decl.hpp"



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

    const std::vector<size_t>& shape() const override { return shape_; }
    const std::vector<size_t>& stride() const override { return stride_; }

    size_t size() const override { return total_size_; }
    
    size_t rank() const override { return rank_; }
    size_t offset() const override { return offset_; }

    //--------------------------------Setter functions
    void set_shape(const std::vector<size_t>& tshape) {
        if (compute_total_size(tshape) != compute_total_size(shape_)) throw std::invalid_argument("Cannot reshape lens: total size mismatch.");
        shape_ = std::move(tshape);
        stride_ = compute_strides(tshape);
        total_size_ = compute_total_size(tshape);
        rank_ = compute_rank(tshape);
    }
    //--------------------------------At()
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
    //-------------------------------Linear At()
    T& at_linear(size_t linear_idx) {
        if (!is_contiguous()) {
            throw std::runtime_error("at_linear called on non-contiguous tensor");
        }

        size_t final_idx = offset_ + linear_idx;

        if (cached_ptr_) {
            return (*cached_ptr_)[final_idx];
        } else {
            auto ps = data_ptr_.lock();
            if (!ps) throw std::runtime_error("Data expired");
            return (*ps)[final_idx];
        }
    }

    const T& at_linear(size_t linear_idx) const {
        if (!is_contiguous()) {
            throw std::runtime_error("at_linear called on non-contiguous tensor");
        }

        size_t final_idx = offset_ + linear_idx;
        if (cached_ptr_) {
            return (*cached_ptr_)[final_idx];
        } else {
            auto ps = data_ptr_.lock();
            if (!ps) throw std::runtime_error("Data expired");
            return (*ps)[final_idx];
        }
    }
    //-------------------------------Operator()
    template<typename... Indices>
    T operator()(Indices... indices) const { // (1,2,3) overload
        static_assert((std::is_convertible_v<Indices, size_t> && ...), "All indices must be size_t");

        std::array<size_t, sizeof...(Indices)> idxs = {static_cast<size_t>(indices)...};

        if (idxs.size() != shape_.size()) {
            throw std::invalid_argument("Index dimension mismatch in lens::operator()");
        }
        size_t flat = offset_;
        for (size_t i = 0; i < idxs.size(); ++i) {
            if (idxs[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds at dimension " + std::to_string(i));
            }
            flat += idxs[i] * stride_[i];
        }
        if (cached_ptr_) {
            return (*cached_ptr_)[flat];
        } else {
            auto ps = data_ptr_.lock();
            if (!ps) throw std::runtime_error("Data expired");
            return (*ps)[flat];
        }
    }

    T operator()(std::initializer_list<size_t> indices) const { // ({1,2,3}) overload
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Index dimension mismatch in lens::operator()");
        }

        size_t flat = offset_;
        size_t i = 0;
        for (auto idx : indices) {
            if (idx >= shape_[i]) {
                throw std::out_of_range("Index out of bounds at dimension " + std::to_string(i));
            }
            flat += idx * stride_[i];
            ++i;
        }

        if (cached_ptr_) {
            return (*cached_ptr_)[flat];
        } else {
            auto ps = data_ptr_.lock();
            if (!ps) throw std::runtime_error("Data expired");
            return (*ps)[flat];
        }
    }
    //-------------------------------Copy()
    tensr::Tensr<T> copy() const {
        std::vector<T> result(total_size_);

        if (is_contiguous()) {
            for (size_t i = 0; i < total_size_; ++i) {
                result[i] = at_linear(i);
            }
        } else {
            for (size_t i = 0; i < total_size_; ++i) {
                auto idx = indexUtils::unflatten_index(i, shape_);
                result[i] = at(idx);
            }
        }

        return tensr::Tensr<T>(shape_, std::move(result));
    }
    //-------------------------------Flatten()
    tensrLens::lens<T> flatten() const {
        std::vector<size_t> flat_shape = { total_size_ };
        std::vector<size_t> flat_stride = { 1 };

        if (!is_contiguous()) {
            throw std::runtime_error("Flatten requires contiguous tensor");
        }
        return lens<T>(data_ptr_.lock(), flat_shape, flat_stride, offset_);
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
    //-------------------------------Cache 
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
        return stride_ == compute_strides(shape_) && offset_ == 0;
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

};

}

