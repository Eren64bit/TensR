Tensr

Tensr is a modern, extensible C++ tensor library for numerical and scientific computing. It provides efficient N-dimensional arrays, zero-copy tensor views, broadcasting, slicing, reshaping, lazy evaluation, and optional CUDA acceleration — all with a clean, type-safe API using modern C++17 features.

Features

N-dimensional tensor support: tensr::Tensr<T>

Zero-copy tensor views: tensrLens::lens<T> for slicing, reshaping, flattening, and broadcasting

NumPy-style broadcasting for elementwise operations

Advanced operations: slicing, transposing, flattening, reshaping, squeeze/unsqueeze

Lazy evaluation engine (tensrLazy) for building expression trees and evaluating on demand

CUDA acceleration (optional): Run operations on GPU via tensrCUDA for +, -, *, /, %

Device memory management: seamless integration with cudaMalloc, cudaMemcpy, and kernel launch

Shape, stride, and offset management

Bounds-checked element access

Data caching for efficient repeated access

Robust error handling and exception-safe API

Modern C++: smart pointers, templates, type traits

Header-only core (CPU): easy to integrate into any project

⚙️ Requirements

C++17 or newer compiler (GCC, Clang, MSVC)

[Optional] CUDA Toolkit (for GPU acceleration)

No external dependencies for CPU-only usage

Build Instructions

CPU-only:

Just include the include/ directory in your project.

g++ -std=c++17 -Iinclude main.cpp -o tensr_test
./tensr_test

With CUDA Support:

You need to separately compile .cu source files using nvcc and link them.

nvcc -c src/tensrCUDA.cu -o build/tensrCUDA.o
ar rcs build/libtensr.a build/tensrCUDA.o

# Then compile your main app
g++ -std=c++17 -Iinclude main.cpp -Lbuild -ltensr -lcudart -o app

For full integration and multiple kernels, a CMake build system is recommended.

Example Usage

#include "include/tensr.hpp"
#include "include/tensrLens.hpp"
#include "include/tensrCUDA.hpp"

int main() {
    tensr::Tensr<float> A({2, 3});
    A.fill(1.0f);

    for (size_t i = 0; i < A.size(); ++i)
        A.at(indexUtils::unflatten_index(i, A.shape())) = float(i + 1);

    A.info();

    // CUDA accelerated addition (if enabled)
    auto B = A;
    auto C = tensrCUDA<float>::add(A, B);
    C.info();

    // Lazy evaluation
    auto expr = tensrLazy::leaf(A) + tensrLazy::leaf(B);
    auto D = tensrLazy::materialize(*expr);
    D.info();
}

API Overview

Component

Purpose

tensr::Tensr<T>

Main tensor class (data + shape + operations)

tensrLens::lens<T>

Zero-copy view class for slicing and reshaping

tensrCUDA<T>

CUDA-accelerated tensor math backend

broadcast::broadcast_to

Apply NumPy-style broadcasting

tensrOps::reshape, squeeze, transpose, slice

Tensor operations

tensrLazy::materialize

Evaluate lazy expressions

operator+, operator-, etc.

Lazy + CUDA fallback operators

at(), shape(), stride(), data()

Tensor introspection and access

File Structure

include/
├── tensr.hpp              # Main tensor class
├── tensrLens.hpp          # Lens (view) support
├── tensrCUDA.hpp          # CUDA API wrapper
├── tensrCUDA.cuh          # CUDA kernel declarations
├── tensrBase.hpp          # Internal base class
├── tensrUtils.hpp         # Tensor utils (shape, flatten)
├── indexUtils.hpp         # Indexing helpers
├── tensrBroadcast.hpp     # Broadcasting logic
├── tensrLazy.hpp          # Lazy evaluation engine
├── tensrOps_decl.hpp      # Operation declarations
├── tensrOps_imp.hpp       # Operation implementations
├── tensrSugar.hpp         # Operator overloads
main.cpp                   # Example usage

Contributing

Pull requests and issues are welcome! Please open an issue for bug reports or feature requests.

License

MIT License

This project is under active development. The API may change.
