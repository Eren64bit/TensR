#pragma once

#include "tensrLens.hpp"
#include "tensr.hpp"
#include "tensrBroadcast.hpp"
#include <functional>
#include <memory>
#include <cmath>
#include <type_traits>
#include "indexUtils.hpp"

namespace tensrLazy {


template<typename T> struct is_tensor_or_lens : std::false_type {};
template<typename T> struct is_tensor_or_lens<tensr::Tensr<T>> : std::true_type {};
template<typename T> struct is_tensor_or_lens<tensrLens::lens<T>> : std::true_type {};

// Base Expression

template<typename T>
class tensrExpr {
public:
    virtual T eval_at(const std::vector<size_t>& idx) const = 0;
    virtual std::vector<size_t> shape() const = 0;
    virtual ~tensrExpr() {}
};

// Leaf (Tensor veya Lens)
template<typename T>
class tensrLeaf : public tensrExpr<T> {
private:
    tensrLens::lens<T> view_;
public:
    tensrLeaf(const tensrLens::lens<T>& lens) : view_(lens) {}
    tensrLeaf(const tensr::Tensr<T>& tensor) : view_(tensor.view()) {}

    const tensrLens::lens<T>& view() const { return view_; }

    T eval_at(const std::vector<size_t>& idx) const override {
        return view_.at(idx);
    }

    std::vector<size_t> shape() const override {
        return view_.shape();
    }
};

// Unary Expression
template<typename T>
class unaryExpr : public tensrExpr<T> {
private:
    std::shared_ptr<tensrExpr<T>> operand_;
    std::function<T(T)> op_;
    std::vector<size_t> shape_;

public:
    unaryExpr(std::shared_ptr<tensrExpr<T>> operand, std::function<T(T)> f)
        : operand_(std::move(operand)), op_(std::move(f)), shape_(operand_->shape()) {}

    T eval_at(const std::vector<size_t>& idx) const override {
        return op_(operand_->eval_at(idx));
    }

    std::vector<size_t> shape() const override {
        return shape_;
    }
};

// Binary Expression
template<typename T>
class binaryExpr : public tensrExpr<T> {
private:
    std::shared_ptr<tensrExpr<T>> lhs_, rhs_;
    std::function<T(T, T)> op_;
    std::vector<size_t> shape_;

public:
    binaryExpr(std::shared_ptr<tensrExpr<T>> lhs, std::shared_ptr<tensrExpr<T>> rhs, std::function<T(T, T)> f)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)), op_(std::move(f)) {
        if (lhs_->shape() != rhs_->shape()) {
            throw std::invalid_argument("Shapes not broadcasted correctly before binaryExpr");
        }
        shape_ = lhs_->shape();
    }

    T eval_at(const std::vector<size_t>& idx) const override {
        return op_(lhs_->eval_at(idx), rhs_->eval_at(idx));
    }

    std::vector<size_t> shape() const override {
        return shape_;
    }
};

// Broadcast Expression
template<typename T>
class broadcastExpr : public tensrExpr<T> {
private:
    std::shared_ptr<tensrExpr<T>> base_;
    std::vector<size_t> shape_, stride_;

public:
    broadcastExpr(const std::shared_ptr<tensrExpr<T>>& base, const std::vector<size_t>& target_shape)
        : base_(base), shape_(target_shape) {
        if (!broadcast::can_broadcast(base_->shape(), shape_)) {
            throw std::runtime_error("broadcastExpr: incompatible shapes");
        }
        stride_ = broadcast::compute_broadcast_stride(base_->shape(), shape_);
    }

    T eval_at(const std::vector<size_t>& idx) const override {
        std::vector<size_t> mapped_idx(base_->shape().size());
        size_t offset = idx.size() - mapped_idx.size();

        for (size_t i = 0; i < mapped_idx.size(); ++i) {
            mapped_idx[i] = (base_->shape()[i] == 1) ? 0 : idx[offset + i];
        }
        return base_->eval_at(mapped_idx);
    }

    std::vector<size_t> shape() const override {
        return shape_;
    }
};

// Materialize Expression

template<typename T>
tensr::Tensr<T> materialize(const tensrExpr<T>& expr) {
    auto shape = expr.shape();
    tensr::Tensr<T> result(shape);
    for (size_t i = 0; i < result.size(); ++i) {
        auto idx = indexUtils::unflatten_index(i, shape);
        result.at(idx) = expr.eval_at(idx);
    }
    return result;
}

// Leaf Helpers
template<typename T>
std::shared_ptr<tensrExpr<T>> leaf(const tensr::Tensr<T>& t) {
    return std::make_shared<tensrLeaf<T>>(t);
}

template<typename T>
std::shared_ptr<tensrExpr<T>> leaf(const tensrLens::lens<T>& l) {
    return std::make_shared<tensrLeaf<T>>(l);
}

// Broadcast Helpers
template<typename T>
std::shared_ptr<tensrExpr<T>> broadcast_to(std::shared_ptr<tensrExpr<T>> base, const std::vector<size_t>& target_shape) {
    return std::make_shared<broadcastExpr<T>>(base, target_shape);
}

// Binary with auto broadcast
template<typename T>
std::shared_ptr<tensrExpr<T>> broadcast_binary(
    std::shared_ptr<tensrExpr<T>> lhs,
    std::shared_ptr<tensrExpr<T>> rhs,
    std::function<T(T, T)> op
) {
    if (lhs->shape() != rhs->shape()) {
        auto target = broadcast::compute_broadcast_shape(lhs->shape(), rhs->shape());
        lhs = broadcast_to(lhs, target);
        rhs = broadcast_to(rhs, target);
    }
    return std::make_shared<binaryExpr<T>>(lhs, rhs, op);
}

// Binary Ops Macro
#define DEFINE_BINARY_OP(opname, opfunc) \
template<typename T> \
std::shared_ptr<tensrExpr<T>> operator opname (std::shared_ptr<tensrExpr<T>> lhs, std::shared_ptr<tensrExpr<T>> rhs) { \
    return broadcast_binary<T>(lhs, rhs, [](T a, T b) { return opfunc; }); \
}

DEFINE_BINARY_OP(+, a + b)
DEFINE_BINARY_OP(-, a - b)
DEFINE_BINARY_OP(*, a * b)
DEFINE_BINARY_OP(/, a / b)
#undef DEFINE_BINARY_OP

// Unary Ops
template<typename T>
std::shared_ptr<tensrExpr<T>> operator-(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return -a; });
}

template<typename T>
std::shared_ptr<tensrExpr<T>> abs(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return std::abs(a); });
}

} // namespace tensrLazy
