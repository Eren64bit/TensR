#pragma once

#include "Tensr.h"

//***************************************************************Operator (+)
template<typename T, DeviceType Device>
Tensr<T, Device> operator+(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal);

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator+(const Tensr<T, Device>& t, Scalar scalar);

//***************************************************************END

//***************************************************************Operator (-)
template<typename T, DeviceType Device>
Tensr<T, Device> operator-(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal);