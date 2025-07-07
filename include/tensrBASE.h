#pragma once

//internal includes
#include "Config.h"


//extarnal includes
#include <memory>
#include <vector>
#include <stdexcept>


enum class backendType {
    CPU, // CPU backend
    GPU // CUDA backend
};

//tensrPolicy
// This enum defines the policy for tensor handling
enum class tensrPolicy {
    VIEW = 'V', // View policy
    TENSOR = 'T' // Tensor policy
};
// Execution modes
// This enum defines the execution modes for the tensor library
// NORMAL: Normal execution mode
// LAZY: Lazy execution mode
// CUDA: CUDA execution mode
// The execution mode can be set using the TENSR_MODE macro defined in Config.h
enum class executionMode { NORMAL, LAZY, CUDA };

//
template<typename Derived, typename T>
class tensrBASE {
    protected:
        std::shared_ptr<std::vector<T>> data_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t offset_;

        tensrPolicy policy_; // 'V' for view, 'T' for tensor
        #if TENSR_MODE == TENSR_MODE_CUDA
            static constexpr executionMode mode_ = executionMode::CUDA; // Default Execution for CUDA mode
        #elif TENSR_MODE == TENSR_MODE_LAZY
            static constexpr executionMode mode_ = executionMode::LAZY; // Default Execution for lazy mode
        #else 
            static constexpr executionMode mode_ = executionMode::NORMAL; // Default Execution for normal mode
        #endif

        

    public:
        // Constructor
        TensrBase() = default;
        virtual ~TensrBase() = default; // virtual destructor for base class

        virtual std::shared_ptr<std::vector<T>> data() const = 0; // pure virtual function to get data pointer
        const std::vector<size_t>& shape() const { return shape_; };
        const std::vector<size_t>& strides() const { return strides_; };
        size_t offset() const { return offset_; };

        virtual size_t size() const {
            size_t total_size = 1;
            for (const auto& dim : shape_) total_size *= dim;
            return total_size;
        }

        // Function to get single data from the tensor or view
        T& at(const std::vector<size_t>& indices) {
            #if TENSR_DEBUG
                if (indices.size() != shape_.size()) throw std::out_of_range("Number of indices does not match tensor dimensions.");
            #endif
                return (*data_)[flatten_index(indices)];
        };
        const T& at(const std::vector<size_t>& indices) const {
            #if TENSR_DEBUG
                if (indices.size() != shape_.size()) throw std::out_of_range("Number of indices does not match tensor dimensions.");
            #endif
                return (*data_)[flatten_index(indices)];
        };

        // Function to reshape the tensor or view
        // This function returns a new instance of the derived class with the new shape
        Derived reshape(const std::vector<size_t>& new_shape) {
            Derived result = static_cast<Derived&>(*this);
            result.shape_ = new_shape;
            result.strides_ = compute_strides(new_shape);
            result.offset_ = 0; // Reset offset for reshaped tensor
            result.data_ = data_; // Keep the same data pointer
            return result;
        }

        // Function to change the tensrPolicy of the tensor or view
        Derived to_mode(tensrPolicy new_policy) {
            if (new_policy == policy_) {
                return static_cast<Derived&>(*this); // No change needed
            }
            Derived result = static_cast<Derived&>(*this);
            result.policy_ = new_policy;
            return result; // Return a new instance with the updated policy
        }

        //helper function to calculate the flattened index from multi-dimensional indices
        size_t flatten_index(const std::vector<size_t>& indices) const {
            size_t idx = offset_;
            for (size_t i = 0; i < indices.size(); ++i) {
                idx += indices[i] * strides_[i];
            }
            return idx;
        }
        // Function to compute strides based on the shape of the tensor
        inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) const {
            std::vector<size_t> strides(shape.size());
            if (shape.empty()) return strides; // Return empty strides for empty shape
            strides.back() = 1; // Last dimension stride is always 1
            for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            return strides;
        }
};


