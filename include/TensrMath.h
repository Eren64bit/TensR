#pragma once
#include "Tensr.h"

#include <functional>
#include <cmath>

template<typename T, DeviceType Device>
Tensr<T, Device>apply_elementwise(const Tensr<T, Device>& t, std::function<T(T)>func) {
    Tensr<T, Device> result(t.shape());
    auto& res_data = result.mutable_data();
    const auto& t_data = t.data();
    for (size_t i = 0; i < t.size(); i++) {
        res_data[i] = func(t_data[i]);
    }

    return result;
}

template<typename T, DeviceType Device>
Tensr<T, Device> exp(const Tensr<T, Device>& t) {
    return apply_elementwise<T, Device>(t, [](T x) { return std::exp(x); });
}

template<typename T, DeviceType Device>
Tensr<T, Device> sigmoid(const Tensr<T, Device>& t) {
    return apply_elementwise<T, Device>(t, [](T x) { return T(1) / (T(1) + std::exp(-x)); });
}

template<typename T, DeviceType Device>
Tensr<T, Device> relu(const Tensr<T, Device>& t) {
    return apply_elementwise<T, Device>(t, [](T x) { return x > T(0) ? x : T(0); });
}

