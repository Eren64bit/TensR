# Tensr

Tensr is a modern, extensible C++ tensor library for numerical computing and machine learning. This version supports **lazy evaluation**, **tensor views** (`TensrLens`), and robust broadcasting, enabling efficient memory usage and high-performance operations on large, multi-dimensional arrays.

## Key Features

- N-dimensional tensor support
- Template-based for any arithmetic type
- Device abstraction (CPU, CUDA planned)
- Shape, stride, and rank management
- NumPy-style broadcasting
- Bounds-checked element access
- Reshape, squeeze, and unsqueeze functionality
- **Lazy evaluation** for efficient computation and memory usage
- **Tensor views** (`TensrLens`) for zero-copy slicing and broadcasting
- Exception-safe API

## Getting Started

### Requirements

- C++17 or later
- (Optional) CUDA Toolkit for CUDA device support

### Building

Tensr is header-only. Just include the headers in your project:

```cpp
#include "tensr/Tensr.hpp"
#include "tensr/Lens.hpp"
#include "tensr/ops/Arithmetic.hpp"
#include "tensr/ops/Broadcast.hpp"
#include "tensr/util/TensrUtils.hpp"
#include "tensr/util/IndexUtils.hpp"
```

### Example Usage

```cpp
#include "tensr/Tensr.hpp"
#include "tensr/Lens.hpp"
#include "tensr/ops/Arithmetic.hpp"

int main() {
    Tensr<float> a({2, 3}); // 2x3 tensor of floats, initialized to zero
    Tensr<float> b({1, 3}, {1, 2, 3}); // 1x3 tensor with data

    // Broadcasting and binary operation (elementwise addition)
    auto c = binary_op<float>(a, b, [](float x, float y) { return x + y; });

    // Tensor view (TensrLens): zero-copy slicing
    TensrLens<float> view = a.slice({0}); // View of the first row

    // Reshape, squeeze, unsqueeze
    a.reshape({3, 2});
    a.squeeze();
    a.unsqueeze(0);

    return 0;
}
```

## API Overview

- `Tensr<T>`: Main tensor class template
- `TensrLens<T>`: Tensor view for zero-copy slicing and broadcasting
- `Tensr<T> binary_op(const TensorA&, const TensorB&, BinaryOp)`: Generic binary operation with broadcasting
- `T& at(const std::vector<size_t>& indices)`: Access element (with bounds checking)
- `void reshape(const std::vector<size_t>& new_shape)`: Change tensor shape (total size must match)
- `void squeeze()`: Remove dimensions of size 1
- `void unsqueeze(int axis)`: Add a dimension of size 1 at the given axis
- `size_t size() const`: Total number of elements
- `size_t rank() const`: Number of dimensions
- `const std::vector<size_t>& shape() const`: Get shape
- `const std::vector<size_t>& stride() const`: Get strides
- `const std::shared_ptr<std::vector<T>>& data() const`: Get raw data pointer

## Lazy Evaluation

All arithmetic operations are evaluated lazily. Computation is only performed when you access the data (e.g., via `at()` or when explicitly requested). This enables efficient chaining of operations and avoids unnecessary intermediate memory allocations.

## Tensor Views (`TensrLens`)

`TensrLens` provides zero-copy views into tensors, allowing efficient slicing, broadcasting, and sub-tensor operations without duplicating data.

## Device Support

- `CPU`: Standard CPU memory
- `CUDA`: (Planned) CUDA device support

## Contributing

Pull requests and issues are welcome! Please open an issue for bug reports or feature requests.

## License

MIT License

---

**Note:** This project is under active development and the API may change.