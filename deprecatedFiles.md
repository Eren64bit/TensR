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
    using T = tensrLazy::tensor_value_type_t<L>;
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
    using T = tensrLazy::tensor_value_type_t<L>;
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
    using ValueT = tensrLazy::tensor_value_type_t<T>;
    auto operand_leaf = leaf(operand);
    return std::make_shared<tensrLazy::unaryExpr<ValueT>>(operand_leaf, [](ValueT a) { return -a; });
}

template<typename T>
auto abs(const T& operand)
    ->  std::enable_if_t<is_tensor_or_lens<std::decay_t<T>>::value,
        std::shared_ptr<tensrLazy::tensrExpr<tensrLazy::tensor_value_type_t<T>>>
    >
{
    using ValueT = tensrLazy::tensor_value_type_t<T>;
    auto operand_leaf = leaf(operand);
    return std::make_shared<tensrLazy::unaryExpr<ValueT>>(operand_leaf, [](ValueT a) { return std::abs(a); });
}

template<typename T>
auto sqrt(const T& operand)
    -> std::enable_if_t<is_tensor_or_lens<std::decay_t<T>>::value,
       std::shared_ptr<tensrLazy::tensrExpr<tensrLazy::tensor_value_type_t<T>>>
       >
{
    using ValueT = tensrLazy::tensor_value_type_t<T>;
    auto operand_leaf = leaf(operand);
    return std::make_shared<tensrLazy::unaryExpr<ValueT>>(operand_leaf, [](ValueT a) { return std::sqrt(a); });
}

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
