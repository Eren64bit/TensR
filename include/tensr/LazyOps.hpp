#pragma once

#include "LazyEval.hpp"
#include "Tensr.hpp"

template <typename T>
std::shared_ptr<TensrExpr<T>> to_expr_ptr(const TensrBase<T>& tensor) {
    return std::make_shared<TensorHolder<T>>(tensor);
}

template <typename T>
std::shared_ptr<TensrExpr<T>> to_expr_ptr(std::shared_ptr<TensrExpr<T>> expr) {
    return expr;
}

template <typename T>
std::shared_ptr<TensrExpr<T>> operator+(
    const TensrBase<T>& a, const TensrBase<T>& b
) {
    auto exprA = to_expr_ptr(a);
    auto exprB = to_expr_ptr(b);
    auto shape = compute_broadcast_shape(a.shape(), b.shape());

    return std::make_shared<BinaryExpr<T, std::plus<T>>>(
        exprA, exprB, std::plus<T>(), shape
    );
}

template <typename T>
std::shared_ptr<TensrExpr<T>> operator-(
    const TensrBase<T>& a, const TensrBase<T>& b
) {
    auto exprA = to_expr_ptr(a);
    auto exprB = to_expr_ptr(b);
    auto shape = compute_broadcast_shape(a.shape(), b.shape());

    return std::make_shared<BinaryExpr<T, std::minus<T>>>(
        exprA, exprB, std::minus<T>(), shape
    );
}

template <typename T>
std::shared_ptr<TensrExpr<T>> operator*(
    const TensrBase<T>& a, const TensrBase<T>& b
) {
    auto exprA = to_expr_ptr(a);
    auto exprB = to_expr_ptr(b);
    auto shape = compute_broadcast_shape(a.shape(), b.shape());

    return std::make_shared<BinaryExpr<T, std::multiplies<T>>>(
        exprA, exprB, std::multiplies<T>(), shape
    );
}

template <typename T>
std::shared_ptr<TensrExpr<T>> operator/(
    const TensrBase<T>& a, const TensrBase<T>& b
) {
    auto exprA = to_expr_ptr(a);
    auto exprB = to_expr_ptr(b);
    auto shape = compute_broadcast_shape(a.shape(), b.shape());

    return std::make_shared<BinaryExpr<T, std::divides<T>>>(
        exprA, exprB, std::divides<T>(), shape
    );
}