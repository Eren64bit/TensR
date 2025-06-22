#pragma once

#include "Tensr.hpp"
#include "Lens.hpp"
#include "util/IndexUtils.hpp"

template <typename T>
struct TensrExpr {
    virtual T eval(const std::vector<size_t>& idx) const = 0;
    virtual const std::vector<size_t> shape() const = 0;
    virtual ~TensrExpr() = default;
};


template <typename T>
class TensorHolder : public TensrExpr<T> {
    const TensrBase<T>& tensor;

public:
    TensorHolder(const TensrBase<T>& t) : tensor(t) {}

    T eval(const std::vector<size_t>& idx) const override {
        return tensor.at(idx);
    }
    const std::vector<size_t> shape() const override {
        return tensor.shape();
    }
    
};

template <typename T, typename OP>
class BinaryExpr : public TensrExpr<T> {
    std::shared_ptr<TensrExpr<T>> left;
    std::shared_ptr<TensrExpr<T>> right;
    OP op;
    std::vector<size_t> shape_;

public:
    BinaryExpr(std::shared_ptr<TensrExpr<T>> a, std::shared_ptr<TensrExpr<T>> b, OP operation, const std::vector<size_t>& shape)
        : left(a), right(b), op(operation), shape_(shape) {}

    T eval(const std::vector<size_t>& idx) const override {
        return op(left->eval(idx), right->eval(idx));
    }

    const std::vector<size_t> shape() const override {
        return shape_;
    }

};

template <typename T>
Tensr<T> evaluate(const std::shared_ptr<TensrExpr<T>>& expr) {
    Tensr<T> result(expr->shape());
    auto stride = compute_strides(expr->shape());
    size_t total_size = compute_total_size(expr->shape());

    for (size_t flat = 0; flat < total_size; ++flat) {
        auto multi_idx = unflaten_index(flat, stride, expr->shape());
        result.at(multi_idx) = expr->eval(multi_idx);
    }

    return result;
}