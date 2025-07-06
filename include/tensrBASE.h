#pragma once

//internal includes
#include "Config.h"

//extarnal includes
#include <memory>
#include <vector>
#include <stdexcept>


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
enum class ExecutionMode { NORMAL, LAZY, CUDA };

//
template<typename T>
class tensrBASE {
    protected:
        std::shared_ptr<std::vector<T>> data_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t offset_;

        tensrPolicy policy_; // 'V' for view, 'T' for tensor
        executionMode mode_ = executionMode::NORMAL; // Execution mode
    public:

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

        T& at(const std::vector<size_t>& indices) = {
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

        //helper function to calculate the flattened index from multi-dimensional indices
        size_t flatten_index(const std::vector<size_t>& indices) const {
            size_t idx = offset_;
            for (size_t i = 0; i < indices.size(); ++i) {
                idx += indices[i] * strides_[i];
            }
            return idx;
        }
};


