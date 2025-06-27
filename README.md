# Tensr

**Tensr** is a modern, extensible C++ tensor library for numerical computing, machine learning, and scientific computing. It provides efficient N-dimensional arrays, zero-copy tensor views, broadcasting, and a clean, type-safe API using modern C++ features.

---

## Features

- **N-dimensional tensor support** (`tensr::Tensr<T>`)
- **Zero-copy tensor views** (`tensrLens::lens<T>`) for slicing, reshaping, and broadcasting
- **NumPy-style broadcasting** for elementwise operations
- **Shape, stride, and offset management**
- **Bounds-checked element access**
- **Reshape, flatten, and transpose operations**
- **Cache mechanism** for efficient repeated access
- **Exception-safe API** with robust error handling
- **Modern C++**: smart pointers, templates, type traits
- **Header-only**: easy to integrate into any project

---

## Getting Started

### Requirements

- C++17 or newer compiler (GCC, Clang, MSVC)
- No external dependencies

### Build

Just include the `include/` directory in your project.  
To build and run the example with g++:

```sh
g++ -std=c++17 -Iinclude main.cpp -o tensr_test
./tensr_test
```

---

## Example Usage

```cpp
#include "include/tensr.hpp"
#include "include/tensrLens.hpp"
#include "include/tensrOps_decl.hpp"
#include "include/tensrOps_imp.hpp"
#include "include/tensrBroadcast.hpp"

int main() {
    // Create a 2x3 tensor and fill it
    std::vector<size_t> shape = {2, 3};
    tensr::Tensr<float> t(shape);
    for (size_t i = 0; i < t.size(); ++i) {
        auto idx = indexUtils::unflatten_index(i, shape);
        t.at(idx) = static_cast<float>(i + 1);
    }

    // Create a lens (view) on the tensor
    auto lens = tensrLens::lens<float>(t.data().lock(), t.shape(), t.stride(), t.offset());
    lens.info();

    // Transpose the lens
    auto transposed = lens.transpose({1, 0});
    transposed.info();

    // Slice the lens
    std::vector<tensrOps::SliceRange> ranges = { {0,2,1}, {1,3,1} };
    auto sliced = lens.slice(ranges);
    sliced.info();

    // Flatten and reshape
    auto flat = lens.flatten();
    flat.info();
    auto reshaped = flat.reshape({2,3});
    reshaped.info();

    // Broadcasting example
    std::vector<size_t> target_shape = {2, 3, 4};
    tensr::Tensr<float> tb({1, 3, 1});
    for (size_t i = 0; i < tb.size(); ++i)
        tb.at(indexUtils::unflatten_index(i, {1,3,1})) = float(i+1);
    auto blens = broadcast::broadcast_to(tb, target_shape);
    blens.info();

    return 0;
}
```

---

## API Overview

- `tensr::Tensr<T>`: Main tensor class
- `tensrLens::lens<T>`: Zero-copy tensor view for slicing, reshaping, broadcasting
- `tensrOps::transpose`, `tensrOps::slice`: Tensor operations
- `broadcast::broadcast_to`: Broadcasting utility
- `T& at(const std::vector<size_t>& indices)`: Bounds-checked element access
- `size_t size() const`: Total number of elements
- `size_t rank() const`: Number of dimensions
- `const std::vector<size_t>& shape() const`: Get shape
- `const std::vector<size_t>& stride() const`: Get strides
- `std::weak_ptr<std::vector<T>> data() const`: Access raw data pointer

---

## File Structure

```
include/
    tensr.hpp
    tensrLens.hpp
    tensrBase.hpp
    tensrUtils.hpp
    indexUtils.hpp
    tensrOps_decl.hpp
    tensrOps_imp.hpp
    tensrBroadcast.hpp
main.cpp
```

---

## Contributing

Pull requests and issues are welcome! Please open an issue for bug reports or feature requests.

---

## License

MIT License

---

**Note:** This project is under active development and the API may change.
