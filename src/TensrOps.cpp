#include "../include/TensrOps.h"


//***************************************************************Operator (+)
template<typename T, DeviceType Device>
Tensr<T, Device> operator+(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal){
    if (lVal.shape().size() != rVal.shape().size()) {
        throw std::runtime_error("error:Unmatched Tensor Shapes");
    }
    Tensr<T, Device> result(lVal.shape());
    auto& res_data = result.mutable_data();
    const auto& l_data = lVal.data();
    const auto& r_data = rVal.data();

    for (size_t i = 0; i < l_data.size(); i++) {
        res_data[i] = l_data[i] + r_data[i];
    }

    return result;
}

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator+(const Tensr<T, Device>& t, Scalar scalar) {
    Tensr<T, Device> result(t.shape());
    auto& res_data = result.mutable_data();
    const auto& t_data = t.data();

    for (size_t i = 0; i < t.size(); i++) {
        res_data[i] = t_data[i] + static_cast<T>(scalar);
    }
    return result;
}

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator+(Scalar scalar, const Tensr<T, Device>& t) {
    return t + scalar;
}
//***************************************************************END

//***************************************************************Operator (-)
template<typename T, DeviceType Device>
Tensr<T, Device> operator-(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal) {
    if (lVal.shape().size() != rVal.shape().size()) {
        throw std::runtime_error("error:Unmatched Tensor Shapes");
    }
    Tensr<T, Device> result(lVal.shape());
    auto& res_data = result.mutable_data();
    const auto& l_data = lVal.data();
    const auto& r_data = rVal.data();

    for (size_t i = 0; i < l_data.size(); i++) {
        res_data[i] = l_data[i] - r_data[i];
    }
    return result;
}

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator-(const Tensr<T, Device>& t, Scalar scalar) {
    Tensr<T, Device> result(t.shape());
    auto& res_data = result.mutable_data();
    const auto& t_data = t.data();

    for (size_t i = 0; i < t.size(); i++) {
        res_data[i] = t_data[i] - scalar;
    }
    return result;
}

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator-(Scalar scalar, const Tensr<T, Device>& t) {
    return t - scalar;
}
//***************************************************************END

//***************************************************************Operator (*)
template<typename T, DeviceType Device>
Tensr<T, Device> operator*(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal) {
    if (lVal.shape().size() != rVal.shape().size()) {
        throw std::runtime_error("error:Unmatched Tensor Shapes");
    }
    Tensr<T, Device> result(lVal.shape());
    auto& res_data = result.mutable_data();
    const auto& l_data = lVal.data();
    const auto& r_data = rVal.data();

    for (size_t i = 0; i < l_data.size(); i++) {
        res_data[i] = l_data[i] * r_data[i];
    }
    return result;
}

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator*(const Tensr<T, Device>& t, Scalar scalar) {
    Tensr<T, Device> result(t.shape());
    auto& res_data = result.mutable_data();
    const auto& t_data = t.data();

    for (size_t i = 0; i < t.size(); i++) {
        res_data[i] = t_data[i] * scalar;
    }
    return result;
}

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator*(Scalar scalar, const Tensr<T, Device>& t) {
    return t * scalar;
}
//***************************************************************END

//***************************************************************Operator (/)
template<typename T, DeviceType Device>
Tensr<T, Device> operator/(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal) {
    if (lVal.shape().size() != rVal.shape().size()) {
        throw std::runtime_error("error:Unmatched Tensor Shapes");
    }
    Tensr<T, Device> result(lVal.shape());
    auto& res_data = result.mutable_data();
    const auto& l_data = lVal.data();
    const auto& r_data = rVal.data();

    for (size_t i = 0; i < l_data.size(); i++) {
        if (r_data[i] == 0) {
            throw std::runtime_error("Cannot divide with zero");
        }
        res_data[i] = l_data[i] / r_data[i];
    }
    return result;
}

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator/(const Tensr<T, Device>& t, Scalar scalar) {
    Tensr<T, Device> result(t.shape());
    auto& res_data = result.mutable_data();
    const auto& t_data = t.data();

    for (size_t i = 0; i < t.size(); i++) {
        if (scalar == 0) {
            throw std::runtime_error("Cannot divide with zero");
        }
        res_data[i] = t_data[i] / scalar;
    }
    return result;
}

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator/(Scalar scalar, const Tensr<T, Device>& t) {
    return t / scalar;
}

//***************************************************************END