
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace tensrCUDA {
inline bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess) && (device_count > 0);
}

class memoryPool {
    private:
        struct memoryBlock // 
        {
            void* block_ptr_ = nullptr;
            size_t size = 0;
            bool in_use = false;
        };

        std::vector<memoryBlock> pool_; // all memory blocks
        size_t total_allocated_ = 0;
        size_t total_used_ = 0;

    public:

        void* allocate(size_t size) {
            memoryBlock* best_fit = nullptr;

            for (auto block : pool_) {
                if (!block.in_use && block.size >= size) {
                    if (!best_fit || block.size < best_fit->size) {
                        best_fit = &block;
                    }
                }
            }
            if (best_fit) {

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
            
            void* new_ptr = nullptr;
            cudaError_t err = cudaMalloc(&new_ptr, size);
            if (err != cudaSuccess) throw std::runtime_error("CUDA malloc failed!");

            pool_.push_back(memoryBlock{new_ptr, size, true});
            total_allocated_ += size;
            total_used_ += size;
            return new_ptr;

        }

        void free(void* ptr); // clear memory block data

        void cleanp(); // clear all blocks

        size_t memory_usage() { return total_used_; }
        size_t memory_allocated() const { return total_allocated_; }

};

}