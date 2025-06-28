#pragma once

#include "tensrLens.hpp"
#include "tensr.hpp"
#include <functional>
#include <type_traits>
#include <memory>
#include <cmath>


// Trait to check if it's a tensor or lens
template<typename T>
struct is_tensor_or_lens : std::false_type {};

template<typename T>
struct is_tensor_or_lens<tensr::Tensr<T>> : std::true_type {};

template<typename T>
struct is_tensor_or_lens<tensrLens::lens<T>> : std::true_type {};

namespace tensrLazy {

// Base expression interface
template<typename T>
class tensrExpr {
public:
    virtual T eval_at(const std::vector<size_t>& idx) const = 0;
    virtual std::vector<size_t> shape() const = 0;
    virtual ~tensrExpr() {}
};

// Leaf expression wrapping a lens
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

// Broadcast utility
template<typename T>
std::shared_ptr<tensrExpr<T>> broadcast_to(const std::shared_ptr<tensrExpr<T>>& expr, const std::vector<size_t>& target_shape) {
    auto leaf_ptr = std::dynamic_pointer_cast<tensrLeaf<T>>(expr);
    if (!leaf_ptr) {
        throw std::runtime_error("Only leaf expressions can be broadcasted in current lazy system");
    }
    const auto& lens_ref = leaf_ptr->view();
    auto broadcasted_lens = broadcast::broadcast_to(lens_ref, target_shape);
    return std::make_shared<tensrLeaf<T>>(broadcasted_lens);
}

// Binary expression
template<typename T>
class binaryExpr : public tensrExpr<T> {
private:
    std::shared_ptr<tensrExpr<T>> lhs_, rhs_;
    std::function<T(T, T)> op;
    std::vector<size_t> shape_;
public:
    binaryExpr(const std::shared_ptr<tensrExpr<T>>& lhs, const std::shared_ptr<tensrExpr<T>>& rhs, std::function<T(T, T)> f)
        : lhs_(lhs), rhs_(rhs), op(std::move(f)) {
        auto lhs_shape = lhs_->shape();
        auto rhs_shape = rhs_->shape();
        if (lhs_shape != rhs_shape) {
            throw std::invalid_argument("Shapes not broadcasted correctly before binaryExpr construction.");
        }
        shape_ = lhs_shape;
    }

    T eval_at(const std::vector<size_t>& idx) const override {
        return op(lhs_->eval_at(idx), rhs_->eval_at(idx));
    }

    std::vector<size_t> shape() const override {
        return shape_;
    }
};

// Unary expression
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

// Materialize expression to a concrete tensor
template<typename T>
tensr::Tensr<T> materialize(const tensrExpr<T>& expr) {
    auto shape = expr.shape();
    tensr::Tensr<T> out(shape);
    for (size_t i = 0; i < out.size(); i++) {
        auto idx = indexUtils::unflatten_index(i, shape);
        out.at(idx) = expr.eval_at(idx);
    }
    return out;
}

// Leaf wrappers
template<typename T>
std::shared_ptr<tensrExpr<T>> leaf(const tensr::Tensr<T>& t) {
    return std::make_shared<tensrLeaf<T>>(t);
}

template<typename T>
std::shared_ptr<tensrExpr<T>> leaf(const tensrLens::lens<T>& l) {
    return std::make_shared<tensrLeaf<T>>(l);
}

// Helper for binary ops with broadcasting
template<typename T>
std::shared_ptr<tensrExpr<T>> broadcast_binary(
    std::shared_ptr<tensrExpr<T>> lhs,
    std::shared_ptr<tensrExpr<T>> rhs,
    std::function<T(T,T)> op
) {
    if (lhs->shape() != rhs->shape()) {
        auto target_shape = broadcast::compute_broadcast_shape(lhs->shape(), rhs->shape());
        lhs = broadcast_to(lhs, target_shape);
        rhs = broadcast_to(rhs, target_shape);
    }
    return std::make_shared<binaryExpr<T>>(lhs, rhs, op);
}

// Binary operators

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

// Unary operators

template<typename T>
std::shared_ptr<tensrExpr<T>> operator-(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return -a; });
}

template<typename T>
std::shared_ptr<tensrExpr<T>> abs(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return std::abs(a); });
}

template<typename T>
std::shared_ptr<tensrExpr<T>> sqrt(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return std::sqrt(a); });
}

template<typename T>
std::shared_ptr<tensrExpr<T>> exp(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return std::exp(a); });
}

template<typename T>
std::shared_ptr<tensrExpr<T>> log(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return std::log(a); });
}

template<typename T>
std::shared_ptr<tensrExpr<T>> sin(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return std::sin(a); });
}

template<typename T>
std::shared_ptr<tensrExpr<T>> cos(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return std::cos(a); });
}

template<typename T>
std::shared_ptr<tensrExpr<T>> tan(std::shared_ptr<tensrExpr<T>> operand) {
    return std::make_shared<unaryExpr<T>>(operand, [](T a) { return std::tan(a); });
}

} // namespace tensrLazy
