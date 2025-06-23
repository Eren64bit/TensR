1. TensorBase<T> (Abstract Base Class)

Responsibility:
Defines the common interface for all tensor-like objects. All tensor types (standard, lens/view, GPU-backed) will inherit from this.

Key Methods (Pure Virtual):

    const std::vector<size_t>& shape() const

    const std::vector<size_t>& stride() const

    T at(const std::vector<size_t>& idx) const

    T& at(const std::vector<size_t>& idx)

    size_t size() const

2. Tensor<T> (Main Tensor Implementation)

Responsibility:
Owns data and manages shape/stride. Represents real, allocated tensors.

Internal State:

    std::vector<size_t> shape_

    std::vector<size_t> stride_

    std::shared_ptr<std::vector<T>> data_

Key Features:

    Contiguous memory layout

    Shape/stride calculations

    Copy/move support

3. TensorLens<T> (View/Reference Tensor)

Responsibility:
Non-owning view into a tensor. Used for slicing, broadcasting, etc.

Internal State:

    std::vector<size_t> shape_

    std::vector<size_t> stride_

    size_t offset_

    std::weak_ptr<std::vector<T>> data_

4. TensorCuda<T> (GPU Tensor, Future Work)

Responsibility:
Stores and manipulates tensors in CUDA device memory.

Notes:

    Needs custom memory allocator

    Unified interface with TensorBase

5. TensorUtils.hpp

Responsibility:
Functional-style utilities for reshaping and modifying tensors.

Functions:

    reshape(tensor, new_shape)

    squeeze(tensor)

    unsqueeze(tensor, axis)

    compute_strides(shape)

    compute_total_size(shape)

6. LensUtils.hpp

Responsibility:
View operations like slicing, transpose, etc.

Functions:

    slice(tensor, axis, start, end)

    transpose(tensor, axes)

    expand_dims(tensor, axis)

7. Broadcasting System

Responsibility:
Handles element-wise operations with different shapes using broadcasting rules.

Operator Overloads:

    operator+

    operator-

    operator*

    operator/

Implementation:

    Lazy wrappers (BroadcastExpr) for shape alignment

    Shape inference and stride masking

8. Lazy Evaluation System (TensorExpr)

Responsibility:
Delays computation until explicitly evaluated. Enables operation chaining without temporary allocations.

Key Types:

    TensorExpr<T> (abstract)

    BinaryExpr<T> (for +, -, etc.)

    ScalarExpr<T> (tensor-scalar ops)

    evaluate(expr) → returns computed Tensor<T>