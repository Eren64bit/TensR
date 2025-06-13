# Tensr

Tensr is a lightweight, header-only C++ tensor library designed for numerical computing and machine learning applications. It supports multi-dimensional arrays (tensors) with customizable data types and device backends (CPU, CUDA).

## Features

- N-dimensional tensor support
- Template-based for arbitrary arithmetic types
- Device abstraction (CPU, CUDA)
- Shape, stride, and rank management
- Bounds-checked element access
- Reshape functionality
- Exception-safe API

## Getting Started

### Requirements

- C++17 or later
- (Optional) CUDA Toolkit for CUDA device support

### Building

Tensr is header-only. Simply include the headers in your project:

```cpp
#include "include/Tensr.h"
```

### Example Usage

```cpp
#include "Tensr.h"

int main() {
    Tensr<float> t({2, 3}); // 2x3 tensor of floats, initialized to zero

    t.at({0, 1}) = 5.0f;    // Set element at (0,1)
    float value = t.at({0, 1}); // Get element at (0,1)

    t.reshape({3, 2});      // Reshape to 3x2

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

## Device Support

- `DeviceType::CPU`: Standard CPU memory
- `DeviceType::CUDA`: (Planned) CUDA device support

## Contributing

Pull requests and issues are welcome! Please open an issue for bug reports or feature requests.

## License

MIT License

---

**Note:** This project is under active development and the API may change.
