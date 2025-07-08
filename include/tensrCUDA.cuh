#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <algorithm>
#include <iostream>

namespace tensrCUDA {

/// controller for checking CUDA availability
inline bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess) && (device_count > 0);
}

// /// Error checking utility for CUDA calls
inline void check_cuda(cudaError_t result, const char* msg = "CUDA error") {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(result));
    }
}


// Memory pool class for managing GPU memory allocation and deallocation
/// This class implements a singleton pattern to ensure only one instance manages the memory pool.
class memoryPool {
private:
    struct memoryBlock {
        void* block_ptr_ = nullptr;
        size_t size = 0;
        bool in_use = false;
    };

    std::vector<memoryBlock> pool_;      // memory pool
    size_t total_allocated_ = 0;         // total memory allocated on GPU
    size_t total_used_ = 0;              // total memory currently in use

    std::mutex pool_mutex;               // mutex for thread safety
    size_t free_count_ = 0;              // count of free operations

    static constexpr size_t COALESCE_THRESHOLD = 10; // threshold for coalescing memory blocks

    // Singleton constructor
    memoryPool() {}
    ~memoryPool() { cleanup(); }

    memoryPool(const memoryPool&) = delete;
    memoryPool& operator=(const memoryPool&) = delete;

public:
    /// Singleton instance getter
    static memoryPool& get_instance() {
        static memoryPool instance;
        return instance;
    }

    /// GPu memory allocation
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(pool_mutex);

        // check for valid size
        memoryBlock* best_fit = nullptr;
        for (auto& block : pool_) {
            if (!block.in_use && block.size >= size) {
                if (!best_fit || block.size < best_fit->size) {
                    best_fit = &block;
                }
            }
        }

        if (best_fit) {
            // use the best fit block
            if (best_fit->size > size * 1.2) {
                memoryBlock new_block;
                new_block.block_ptr_ = static_cast<char*>(best_fit->block_ptr_) + size;
                new_block.size = best_fit->size - size;
                new_block.in_use = false;
                best_fit->size = size;
                pool_.push_back(new_block);
            }
            best_fit->in_use = true;
            total_used_ += best_fit->size;
            return best_fit->block_ptr_;
        }

        // no suitable block found, allocate new memory
        void* new_ptr = nullptr;
        check_cuda(cudaMalloc(&new_ptr, size), "cudaMalloc failed");
        pool_.push_back(memoryBlock{new_ptr, size, true});
        total_allocated_ += size;
        total_used_ += size;
        return new_ptr;
    }

    /// Free allocated memory
    void free(void* ptr) {
        std::lock_guard<std::mutex> lock(pool_mutex);

        for (auto& block : pool_) {
            if (block.block_ptr_ == ptr) {
                if (!block.in_use) {
                    throw std::runtime_error("Double free detected!");
                }
                block.in_use = false;
                total_used_ -= block.size;

                // cudaFree(block.block_ptr_);
                if (++free_count_ % COALESCE_THRESHOLD == 0) {
                    coalesce();
                }
                return;
            }
        }
        throw std::runtime_error("Invalid free: Pointer not found in memory pool");
    }

    /// Coalesce adjacent free memory blocks
    void coalesce() {
        std::lock_guard<std::mutex> lock(pool_mutex);

        if (pool_.empty()) return;

        // sort the pool by block pointer
        std::sort(pool_.begin(), pool_.end(),
            [](const memoryBlock& a, const memoryBlock& b) {
                return a.block_ptr_ < b.block_ptr_;
            });

        for (size_t i = 0; i < pool_.size() - 1; ) {
            auto& current = pool_[i];
            auto& next = pool_[i + 1];

            if (!current.in_use && !next.in_use &&
                static_cast<char*>(current.block_ptr_) + current.size == next.block_ptr_) {
                current.size += next.size;
                pool_.erase(pool_.begin() + i + 1);
            } else {
                ++i;
            }
        }
    }

    // 
    void cleanup() {
        std::lock_guard<std::mutex> lock(pool_mutex);

        for (auto& block : pool_) {
            if (block.block_ptr_) {
                cudaFree(block.block_ptr_);
            }
        }
        pool_.clear();
        total_allocated_ = 0;
        total_used_ = 0;
        free_count_ = 0;
    }

    /// information getters
    size_t memory_usage() const { return total_used_; }
    size_t memory_allocated() const { return total_allocated_; }
};

} // namespace tensrCUDA
