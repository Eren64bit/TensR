# Tensr

Tensr is a modern, extensible C++ tensor library designed for numerical computing and machine learning. This new version introduces **lazy evaluation** and **tensor views** (`TensrLens`), enabling efficient memory usage and high-performance operations on large, multi-dimensional arrays.

## Key Features

- N-dimensional tensor support
- Template-based for any arithmetic type
- Device abstraction (CPU, CUDA planned)
- Shape, stride, and rank management
- Broadcasting (NumPy-style)
- Bounds-checked element access
- Reshape and slicing functionality
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
#include "tensr/TensrLens.hpp"
#include "tensr/TensrOps.hpp"
#include "tensr/TensrMath.hpp"
```

### Example Usage

```cpp
#include "tensr/Tensr.hpp"
#include "tensr/TensrLens.hpp"

int main() {
    Tensr<float> a({2, 3}); // 2x3 tensor of floats, initialized to zero

    // Lazy evaluation: operations are not computed until needed
    auto b = a + 1.0f;      // No computation yet
    float x = b.at({0, 1}); // Computation happens here

    // Tensor view (TensrLens): zero-copy slicing
    TensrLens<float> view = a.slice({0}); // View of the first row

    // Broadcasting with lazy evaluation
    Tensr<float> c({1, 3}, {1, 2, 3});
    auto d = a + c; // Broadcasted addition, computed lazily

    return 0;
}
```

## API Overview

- `Tensr<T>`: Main tensor class template
- `TensrLens<T>`: Tensor view for zero-copy slicing and broadcasting
- `Tensr<T> operator+(const Tensr<T>&, const Tensr<T>&)`: Lazy, broadcasted addition
- `Tensr<T> operator+(const Tensr<T>&, Scalar)`: Lazy, broadcasted addition with scalar
- `T& at(const std::vector<size_t>& indices)`: Access element (with bounds checking)
- `void reshape(std::vector<size_t> new_shape)`: Change tensor shape (total size must match)
- `size_t size() const`: Total number of elements
- `size_t rank() const`: Number of dimensions
- `const std::vector<size_t>& shape() const`: Get shape
- `const std::shared_ptr<T>& data() const`: Get raw data pointer

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
