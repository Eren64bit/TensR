#include "../include/TensrOps.h"

//***************************************************************Transpose
template<typename T, DeviceType Device>
Tensr<T, Device> transpose(const Tensr<T, Device>& t, const std::vector<size_t>& permute) {
    if (permute.size() > t.rank()) {
        throw std::runtime_error("Permute Cannot be bigger than Tensor shape");
    }
    for (size_t i = 0; i < permute.size() - 1; i++) {
        if (permute[i] > t.shape().size() - 1) {
            throw std::runtime_error("Index out of range");
        }
    }
    std::vector<size_t> res_shape;
    for (int i = 0; i < permute.size(); i++) {
        res_shape = t.shape()[permute[i]];
    }

    std::vector<T> new_data(t.size());
    for (int i = 0; i < t.size(); ++i) {
        std::vector<size_t> old_multi_idx = t.unflatten_index_(i);

        std::vector<size_t> new_multi_idx;
        for (size_t p : permute) {
            new_multi_idx.push_back(old_multi_idx[p]);
        }

        size_t new_flat_idx = flatten_index(new_multi_idx, res_shape);
        new_data[new_flat_idx] = t.data()[i];
    }

    return Tensr<T, Device>(res_shape, new_data);
}

template<typename T, DeviceType Device>
Tensr<T, Device> transpose(const Tensr<T, Device>& t) {
    if (t.rank() != 2) {
        throw std::runtime_error("transpose() overload only supports 2D tensors. Use transpose(t, permute) for higher dimensions.");
    }
    return transpose(t, {1, 0});

}
//****************************************************************END
//***************************************************************Squeeze 
template<typename T, DeviceType Device>
Tensr<T, Device> squeeze(const Tensr<T, Device>& t) {
    std::vector<T> dat = t.data();
    std::vector<size_t> res_shape
    for (int i = 0; i < t.shape().size(); i++) {
        if (t.shape()[i] != 1) {
            res_shape.push_back(t.shape()[i]);
        }
    }

    if (res_shape == t.shape()) {
        return t;
    }

    return t.reshape(res_shape);
}

template<typename T, DeviceType Device>
Tensr<T, Device> squeeze(const Tensr<T, Device>& t, int idx) {    
    if (idx >= t.shape().size()) {
        throw std::runtime_error("Index out of range.");
    }
    if (t.shape()[idx] != 1) {
        throw std::runtime_error("Value at the index must be 1");
    } 
    
    std::vector<size_t> res_shape;

    for (size_t i = 0; i < t.shape().size(); i++) {
        if (i != idx) {
            res_shape.push_back(t.shape()[i]);
        }
    }

    return t.reshape(res_shape);

}   
//***************************************************************END

//***************************************************************Unsqueeze

//***************************************************************END

//***************************************************************Operator (+)
template<typename T, DeviceType Device>
Tensr<T, Device> operator+(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal){
    if (lVal.shape().size() != rVal.shape().size()) {
        throw std::runtime_error("error:Unmatched Tensor Shapes\n");
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