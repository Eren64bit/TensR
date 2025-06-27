
#include "tensrLens.hpp"
#include "tensr.hpp"
#include <functional>
#include <type_traits>

template<typename T>
struct is_tensor_or_lens : std::false_type {};

template<typename T>
struct is_tensor_or_lens<tensr::Tensr<T>> : std::true_type {};

template<typename T>
struct is_tensor_or_lens<tensrLens::lens<T>> : std::true_type {};

namespace tensrLazy {

template<typename T>
struct tensor_value_type;

template<typename T>
struct tensor_value_type<tensr::Tensr<T>> {
    using type = T;
};

template<typename T>
struct tensor_value_type<tensrLens::lens<T>> {
    using type = T;
};

template<typename T>
using tensor_value_type_t = typename tensor_value_type<std::decay_t<T>>::type;


template<typename T>
class tensrExpr {
public:
    virtual T eval_at(const std::vector<size_t>& idx) const = 0;
    virtual std::vector<size_t> shape() const = 0;
    virtual ~tensrExpr() {}
};

template<typename T>
class tensrLeaf : public tensrExpr<T> {
private:
    tensrLens::lens<T> view_;
public:

    tensrLeaf(const tensrLens::lens<T>& lens) : view_(std::move(lens)) {}
    tensrLeaf(const tensr::Tensr<T>& tensor) : view_(std::move(tensor.view())) {}

    T eval_at(const std::vector<size_t>& idx) const override {
        return view_.at(idx);
    }

    std::vector<size_t> shape() const override {
        return view_.shape();
    }
};

template<typename T>
class binaryExpr : public tensrExpr<T> {
private:
    std::shared_ptr<tensrExpr<T>> lhs_, rhs_;
    std::function<T(T, T)> op;
    std::vector<size_t> shape_;
public:
    binaryExpr(const std::shared_ptr<tensrExpr<T>>& lhs, const std::shared_ptr<tensrExpr<T>>& rhs, std::function<T(T, T)> f) : lhs_(std::move(lhs)), rhs_(std::move(rhs)), op(std::move(f)){
        auto lhs_shape = lhs_->shape();
        auto rhs_shape = rhs_->shape();
        if (lhs_shape != rhs_shape) {
            throw std::invalid_argument("Shape mismatch in binary operation");
        }

        shape_ = std::move(lhs_shape);
    }

    T eval_at(const std::vector<size_t>& idx) const override {
        return op(lhs_->eval_at(idx), rhs_->eval_at(idx));
    }

    std::vector<size_t> shape() const override {
        return lhs_->shape();
    }
};

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

template<typename T>
std::shared_ptr<tensrLazy::tensrExpr<T>> leaf(const tensr::Tensr<T>& t) {
    return std::make_shared<tensrLazy::tensrLeaf<T>>(t);
}

template<typename T>
std::shared_ptr<tensrLazy::tensrExpr<T>> leaf(const tensrLens::lens<T>& l) {
    return std::make_shared<tensrLazy::tensrLeaf<T>>(l);
}

}

template<typename L, typename R>
auto operator+(const L& left, const R& right)
    -> std::enable_if_t<
        is_tensor_or_lens<std::decay_t<L>>::value &&
        is_tensor_or_lens<std::decay_t<R>>::value,
        std::shared_ptr<tensrLazy::tensrExpr<tensrLazy::tensor_value_type_t<L>>>
    >
{
    using T = tensrLazy::tensor_value_type_t<L>;
    auto lhs_leaf = tensrLazy::leaf(left);
    auto rhs_leaf = tensrLazy::leaf(right);
    return std::make_shared<tensrLazy::binaryExpr<T>>(lhs_leaf, rhs_leaf, [](T a, T b) { return a + b; });
}

template<typename L, typename R>
auto operator-(const L& left, const R& right)
    -> std::enable_if_t<
        is_tensor_or_lens<std::decay_t<L>>::value &&
        is_tensor_or_lens<std::decay_t<R>>::value,
        std::shared_ptr<tensrLazy::tensrExpr<tensrLazy::tensor_value_type_t<L>>>
    >
{
    using T = tensor_value_type_t<L>;
    auto lhs_leaf = leaf(left);
    auto rhs_leaf = leaf(right);
    return std::make_shared<tensrLazy::binaryExpr<T>>(lhs_leaf, rhs_leaf, [](T a, T b) { return a - b; });
}

template<typename L, typename R>
auto operator*(const L& left, const R& right)
    -> std::enable_if_t<
        is_tensor_or_lens<std::decay_t<L>>::value &&
        is_tensor_or_lens<std::decay_t<R>>::value,
        std::shared_ptr<tensrLazy::tensrExpr<tensrLazy::tensor_value_type_t<L>>>
    >
{
    using T = tensor_value_type_t<L>;
    auto lhs_leaf = leaf(left);
    auto rhs_leaf = leaf(right);
    return std::make_shared<tensrLazy::binaryExpr<T>>(lhs_leaf, rhs_leaf, [](T a, T b) { return a * b; });
}

template<typename T>
auto operator-(const T& operand)
    -> std::enable_if_t<is_tensor_or_lens<std::decay_t<T>>::value,
        std::shared_ptr<tensrLazy::tensrExpr<tensrLazy::tensor_value_type_t<T>>>
    >
{
    using ValueT = tensor_value_type_t<T>;
    auto operand_leaf = leaf(operand);
    return std::make_shared<tensrLazy::unaryExpr<ValueT>>(operand_leaf, [](ValueT a) { return -a; });
}

template<typename T>
auto abs(const T& operand)
    ->  std::enable_if_t<is_tensor_or_lens<std::decay_t<T>>::value,
        std::shared_ptr<tensrLazy::tensrExpr<tensrLazy::tensor_value_type_t<T>>>
    >
{
    using ValueT = tensor_value_type_t<T>;
    auto operand_leaf = leaf(operand);
    return std::make_shared<tensrLazy::unaryExpr<ValueT>>(operand_leaf, [](ValueT a) { return std::abs(a); });
}

template<typename T>
auto sqrt(const T& operand)
    -> std::enable_if_t<is_tensor_or_lens<std::decay_t<T>>::value,
       std::shared_ptr<tensrLazy::tensrExpr<tensrLazy::tensor_value_type_t<T>>>
       >
{
    using ValueT = tensor_value_type_t<T>;
    auto operand_leaf = leaf(operand);
    return std::make_shared<tensrLazy::unaryExpr<ValueT>>(operand_leaf, 
                                              [](ValueT a) { return std::sqrt(a); });
}



//return tensrLazy::binaryExpr<T>(std::make_shared<tensrLazy::tensrExpr<T>>(left), std::make_shared<tensrLazy::tensrExpr<T>>(right), [](T a, T b) { return a + b; });