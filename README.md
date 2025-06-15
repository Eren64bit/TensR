# Tensr

Tensr is a lightweight, header-only C++ tensor library for numerical computing and machine learning. It supports N-dimensional arrays (tensors) with customizable data types and device backends (CPU, CUDA). The library provides basic tensor operations, broadcasting, shape and stride management, and safe element access.

## Features

- N-dimensional tensor support
- Template-based for any arithmetic type
- Device abstraction (CPU, CUDA)
- Shape, stride, and rank management
- Broadcasting support (NumPy-style)
- Bounds-checked element access
- Reshape functionality
- Basic arithmetic operations (tensor-tensor and tensor-scalar addition)
- Exception-safe API

## Getting Started

### Requirements

- C++17 or later
- (Optional) CUDA Toolkit for CUDA device support

### Building

Tensr is header-only. Just include the headers in your project:

```cpp
#include "include/Tensr.h"
#include "include/TensrOps.h"
#include "include/TensrBroadcast.h"
#include "include/TensrMath.h"
```

### Example Usage

```cpp
#include "Tensr.h"
#include "TensrOps.h"
#include "TensrBroadcast.h"
#include "TensrMath.h"

int main() {
    Tensr<float> a({2, 3}); // 2x3 tensor of floats, initialized to zero
    Tensr<float> b({1, 3}, {1, 2, 3}); // 1x3 tensor with data

    auto shape = broadcast_shapes(a, b); // Get broadcasted shape
    auto a_bc = broadcast_to(a, shape);  // Broadcast a to shape
    auto b_bc = broadcast_to(b, shape);  // Broadcast b to shape

    auto c = a_bc + b_bc;                // Tensor-tensor addition (broadcasted)
    auto d = b + 10.0f;                  // Tensor-scalar addition

    a.at({0, 1}) = 5.0f;                 // Set element at (0,1)
    float value = a.at({0, 1});          // Get element at (0,1)

    a.reshape({3, 2});                   // Reshape to 3x2

    auto s = sigmoid(a);                 // Elementwise sigmoid

    return 0;
}
```

## API Overview

- `Tensr<T, DeviceType Device = DeviceType::CPU>`: Main tensor class template
- `Tensr(std::vector<size_t> shape)`: Construct tensor with given shape, zero-initialized
- `Tensr(std::vector<size_t> shape, std::vector<T> data)`: Construct tensor with shape and data
- `T& at(const std::vector<size_t>& indices)`: Access element (with bounds checking)
- `void reshape(std::vector<size_t> new_shape)`: Change tensor shape (total size must match)
- `size_t size() const`: Total number of elements
- `size_t rank() const`: Number of dimensions
- `const std::vector<size_t>& shape() const`: Get shape
- `const std::vector<T>& data() const`: Get raw data
- `Tensr<T, Device> operator+(const Tensr<T, Device>&, const Tensr<T, Device>&)`: Tensor-tensor addition
- `Tensr<T, Device> operator+(const Tensr<T, Device>&, Scalar)`: Tensor-scalar addition
- `std::vector<size_t> broadcast_shapes(const Tensr<T, Device>&, const Tensr<T, Device>&)`: Get broadcasted shape
- `Tensr<T, Device> broadcast_to(Tensr<T, Device>&, std::vector<size_t>)`: Broadcast tensor to shape
- `Tensr<T, Device> sigmoid(const Tensr<T, Device>&)`: Elementwise sigmoid

## Device Support

- `DeviceType::CPU`: Standard CPU memory
- `DeviceType::CUDA`: (Planned) CUDA device support

## Contributing

Pull requests and issues are welcome! Please open an issue for bug reports or feature requests.

## License

MIT License

---

**Note:** This project is under active development and the API may change.