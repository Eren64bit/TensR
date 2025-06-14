#pragma once

#include "Tensr.h"

template<typename T, DeviceType Device>
Tensr<T, Device> operator+(const Tensr<T, Device>& lVal, const Tensr<T, Device>& rVal);
