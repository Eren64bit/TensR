# Tensr

**Tensr** is a modern, extensible C++ tensor library for numerical and scientific computing. It provides efficient N-dimensional arrays, zero-copy tensor views, broadcasting, slicing, reshaping, lazy evaluation, and more, all with a clean, type-safe API using modern C++17 features. CUDA acceleration is available for fast elementwise operations on supported hardware.


---

## Features

- **N-dimensional tensor support** (`tensr::Tensr<T>`)
- **Zero-copy tensor views** (`tensrLens::lens<T>`) for slicing, reshaping, flattening, and broadcasting
- **NumPy-style broadcasting** for elementwise operations
- **Advanced slicing, transposing, flattening, reshaping, squeeze/unsqueeze**
- **Lazy evaluation** (`tensrLazy`) for efficient expression trees and deferred computation
- **Shape, stride, and offset management**
- **Bounds-checked element access**
- **Cache mechanism** for efficient repeated access
- **Exception-safe API** with robust error handling
- **Modern C++**: smart pointers, templates, type traits
- **Header-only**: easy to integrate into any project
- **CUDA acceleration** for elementwise operations (`int`, `float`, `double`)

---

## Getting Started

### Requirements

- C++17 or newer compiler (GCC, Clang, MSVC)
- No external dependencies
- CUDA Toolkit and compatible NVIDIA GPU for CUDA suppor

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
#include "include/tensrLazy.hpp"
#include "include/tensrSugar.hpp"

int main() {
    // 1. Create and fill a tensor
    std::vector<size_t> shape = {2, 3};
    tensr::Tensr<float> t(shape);
    t.fill(1.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        auto idx = indexUtils::unflatten_index(i, shape);
        t.at(idx) = static_cast<float>(i + 1);
    }
    t.info();

    // 2. Create a lens (view) and access elements
    auto locked_data = t.data().lock();
    tensrLens::lens<float> l(locked_data, t.shape(), t.stride(), t.offset());
    l.info();
    std::cout << "Lens l(1,2): " << l(1,2) << std::endl;
    std::cout << "Lens l.at({0,1}): " << l.at({0,1}) << std::endl;

    // 3. Transpose, slice, flatten, reshape
    auto lT = l.transpose({1,0});
    lT.info();
    std::vector<tensrOps::SliceRange> ranges = { {0,2,1}, {1,3,1} };
    auto lS = l.slice(ranges);
    lS.info();
    auto lF = l.flatten();
    lF.info();
    auto lR = lF.reshape({2,3});
    lR.info();

    // 4. Copy and fill
    auto t2 = l.copy();
    t2.info();
    t2.fill(42.0f);
    std::cout << "After fill, t2.at({0,0}): " << t2.at({0,0}) << std::endl;

    // 5. Broadcasting
    std::vector<size_t> bshape = {2, 3, 4};
    tensr::Tensr<float> tb({1,3,1});
    for (size_t i = 0; i < tb.size(); ++i)
        tb.at(indexUtils::unflatten_index(i, {1,3,1})) = float(i+1);
    auto blens = broadcast::broadcast_to(tb, bshape);
    blens.info();
    std::cout << "Broadcasted lens (1,2,3): " << blens(1,2,3) << std::endl;

    // 6. Squeeze and Unsqueeze
    tensr::Tensr<float> t_sq({1,2,3});
    auto sq = t_sq.squeeze();
    sq.info();
    auto usq = sq.unsqueeze(0);
    usq.info();

    // 7. Cache test
    l.cache_data_ptr();
    l.info();
    l.clear_cache();

    // 8. Lazy evaluation (tensrLazy)
    tensr::Tensr<float> t3(shape);
    for (size_t i = 0; i < t3.size(); ++i) {
        auto idx = indexUtils::unflatten_index(i, shape);
        t3.at(idx) = static_cast<float>(10 * (i + 1));
    }
    auto expr = tensrLazy::leaf(t) + tensrLazy::leaf(t3);
    auto result = tensrLazy::materialize(*expr);
    result.info();
    std::cout << "Lazy add result (1,2): " << result.at({1,2}) << std::endl;

    // 9. Sugar operators
    auto sum = t + t3;
    sum.info();
    auto diff = t - t3;
    diff.info();
    auto mul = t * t3;
    mul.info();
    auto div = t3 / t;
    div.info();

    std::vector<size_t> shape = {1000, 1000};
    tensr::Tensr<float> a(shape, 1.0f);
    tensr::Tensr<float> b(shape, 2.0f);

    // Elementwise addition on GPU
    auto c = tensrCUDA::tensrCUDA<float>::add(a, b);

    // Elementwise subtraction on GPU
    auto d = tensrCUDA::tensrCUDA<float>::subtract(a, b);

    // Elementwise multiplication on GPU
    auto e = tensrCUDA::tensrCUDA<float>::multiply(a, b);

    // You can access results as usual
    std::cout << "c(0,0): " << c.at({0,0}) << std::endl;

    // 10. Error handling
    try {
        l(5,0); // Out of bounds
    } catch (const std::exception& e) {
        std::cout << "Expected error (out of bounds): " << e.what() << std::endl;
    }
    try {
        l.reshape({4,2}); // Invalid reshape
    } catch (const std::exception& e) {
        std::cout << "Expected error (reshape): " << e.what() << std::endl;
    }

    return 0;
}
```

---

## API Overview

- `tensr::Tensr<T>`: Main tensor class
- `tensrLens::lens<T>`: Zero-copy tensor view for slicing, reshaping, broadcasting
- `tensrOps::transpose`, `tensrOps::slice`, `tensrOps::reshape`, `tensrOps::squeeze`, `tensrOps::unsqueeze`: Tensor operations
- `broadcast::broadcast_to`: Broadcasting utility
- `tensrLazy::materialize`, `operator+`, `operator-`, `operator*`, `operator/`: Lazy operations and materialization
- `T& at(const std::vector<size_t>& indices)`: Bounds-checked element access
- `size_t size() const`: Total number of elements
- `size_t rank() const`: Number of dimensions
- `const std::vector<size_t>& shape() const`: Get shape
- `const std::vector<size_t>& stride() const`: Get strides
- `std::weak_ptr<std::vector<T>> data() const`: Access raw data pointer
- `void fill(const T&)`: Fill tensor or lens with a value
- `void info() const`: Print tensor/lens metadata

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
    tensrLazy.hpp
    tensrSugar.hpp
    CUDA/
        tensrCUDA.h
        tensrCUDA.cuh
        tensrCUDA.cu
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
