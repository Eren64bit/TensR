#pragma once

//internal includes
#include "Config.h"

//external includes
#include <memory>
#include <vector>
#include <stdexcept>

namespace tensrCUDA {
    inline bool cuda_available();
    class memoryPool; // Forward declaration
}

enum class backendType { CPU, GPU };
enum class tensrPolicy { VIEW = 'V', TENSOR = 'T' };
enum class executionMode { NORMAL, LAZY, CUDA };

template<typename Derived, typename T>
class tensrBASE {
protected:
    std::shared_ptr<std::vector<T>> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_ = 0;
    void* gpu_ptr_ = nullptr;

    tensrPolicy policy_; // VIEW or TENSOR

    #if TENSR_MODE == TENSR_MODE_CUDA
        static constexpr executionMode mode_ = executionMode::CUDA;
    #elif TENSR_MODE == TENSR_MODE_LAZY
        static constexpr executionMode mode_ = executionMode::LAZY;
    #else 
        static constexpr executionMode mode_ = executionMode::NORMAL;
    #endif

    backendType backend_;

public:
    TensrBase() {
        backend_ = tensrCUDA::cuda_available() ? backendType::GPU : backendType::CPU;
    }

    virtual ~TensrBase() {
        if (gpu_ptr_) {
            tensrCUDA::memoryPool::get_instance().free(gpu_ptr_);
        }
    }

    virtual std::shared_ptr<std::vector<T>> data() const = 0;
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t offset() const { return offset_; }

    size_t size() const {
        size_t total = 1;
        for (auto dim : shape_) total *= dim;
        return total;
    }

    T& at(const std::vector<size_t>& indices) {
        return (*data_)[flatten_index(indices)];
    }
    const T& at(const std::vector<size_t>& indices) const {
        return (*data_)[flatten_index(indices)];
    }

    Derived reshape(const std::vector<size_t>& new_shape) {
        Derived result = static_cast<Derived&>(*this);
        result.shape_ = new_shape;
        result.strides_ = compute_strides(new_shape);
        return result;
    }

    void allocate_gpu(bool copy_from_cpu = true) {
        if (backend_ != backendType::GPU)
            throw std::runtime_error("Not a GPU backend.");
        gpu_ptr_ = tensrCUDA::memoryPool::get_instance().allocate(size() * sizeof(T));
    }

    void free_gpu() {
        if (backend_ != backendType::GPU)
            throw std::runtime_error("Not a GPU backend.");
        if (gpu_ptr_) {
            tensrCUDA::memoryPool::get_instance().free(gpu_ptr_);
            gpu_ptr_ = nullptr;
        }
    }

    void set_backend(backendType backend);

    size_t flatten_index(const std::vector<size_t>& indices) const {
        size_t idx = offset_;
        for (size_t i = 0; i < indices.size(); ++i) {
            idx += indices[i] * strides_[i];
        }
        return idx;
    }

    std::vector<size_t> compute_strides(const std::vector<size_t>& shape) const {
        std::vector<size_t> strides(shape.size());
        if (!shape.empty()) {
            strides.back() = 1;
            for (int i = shape.size() - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        return strides;
    }
};
