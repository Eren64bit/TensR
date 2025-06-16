#pragma once

#include "Tensr.h"

//***************************************************************Transpose
template<typename T, DeviceType Device>
Tensr<T, Device> transpose(const Tensr<T, Device>& t, const std::vector<size_t>&);
//***************************************************************END

//***************************************************************Squeeze 
template<typename T, DeviceType Device>
Tensr<T, Device> squeeze(const Tensr<T, Device>& t);

template<typename T, DeviceType Device>
Tensr<T, Device> squeeze(const Tensr<T, Device>& t, size_t idx);
//***************************************************************END

//***************************************************************Unsqueeze
template<typename T, DeviceType Device>
Tensr<T, Device> unsquezee();
//***************************************************************END

//***************************************************************Operator (+)
template<typename T, DeviceType Device>
Tensr<T, Device> operator+(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal);

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator+(const Tensr<T, Device>& t, Scalar scalar);

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator+(Scalar scalar, const Tensr<T, Device>& t) ;

//***************************************************************END

//***************************************************************Operator (-)
template<typename T, DeviceType Device>
Tensr<T, Device> operator-(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal);

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator-(const Tensr<T, Device>& t, Scalar scalar);

//***************************************************************END

//***************************************************************Operator (*)
template<typename T, DeviceType Device>
Tensr<T, Device> operator*(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal);

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator*(const Tensr<T, Device>& t, Scalar scalar);

//***************************************************************END

//***************************************************************Operator (/)
template<typename T, DeviceType Device>
Tensr<T, Device> operator/(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal);

template<typename T,DeviceType Device, typename Scalar, typename = std::enable_if_t<std::is_convertible_v<Scalar, T>>>
Tensr<T, Device> operator/(const Tensr<T, Device>& t, Scalar scalar);


//***************************************************************END