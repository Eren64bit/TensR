
#include <cuda_runtime.h>
#include <vector>

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

            
        }

        void free(void* ptr); // clear memory block data

        void cleanp(); // clear all blocks

        size_t memory_usage() { return total_used_; }
        size_t memory_allocated() const { return total_allocated_; }

};

}