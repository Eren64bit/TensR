
#include <stdio.h>
#include <stdlib.h>

extern "C"
{

bool cuda_available()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        printf("CUDA Error in cuda_available: %s\n", cudaGetErrorString(err));
        return false;
    }
    printf("Found %d CUDA device(s)\n", device_count);
    return device_count > 0;
}

void cuda_check(cudaError_t result, const char *msg)
{
    if (result != cudaSuccess)
    {
        printf("CUDA Error: %s - %s\n", msg, cudaGetErrorString(result));
        fflush(stdout);
        exit(EXIT_FAILURE);
    }
}

void *cuda_malloc(size_t size)
{
    if (size == 0) {
        printf("Warning: Attempting to allocate 0 bytes\n");
        return nullptr;
    }
    
    void *new_ptr = nullptr;
    cudaError_t err = cudaMalloc(&new_ptr, size);
    cuda_check(err, "cudaMalloc in cuda_malloc");
    
    printf("Allocated %zu bytes at GPU address %p\n", size, new_ptr);
    return new_ptr;
}

void cuda_free(void *ptr)
{
    if (ptr != nullptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            printf("CUDA Error in cuda_free: %s\n", cudaGetErrorString(err));
        }
    }
}

void cuda_sync()
{
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void cuda_copy_to_gpu(void *dest_ptr, const void *src_ptr, size_t count)
{
    if (dest_ptr == nullptr) {
        printf("Error: dest_ptr is null in cuda_copy_to_gpu\n");
        exit(EXIT_FAILURE);
    }
    if (src_ptr == nullptr) {
        printf("Error: src_ptr is null in cuda_copy_to_gpu\n");
        exit(EXIT_FAILURE);
    }
    cuda_check(cudaMemcpy(dest_ptr, src_ptr, count, cudaMemcpyHostToDevice), "cuda_copy_to_gpu");
}

void cuda_copy_to_cpu(void *dest_ptr, const void *src_ptr, size_t count)
{
    if (dest_ptr == nullptr) {
        printf("Error: dest_ptr is null in cuda_copy_to_cpu\n");
        exit(EXIT_FAILURE);
    }
    if (src_ptr == nullptr) {
        printf("Error: src_ptr is null in cuda_copy_to_cpu\n");
        exit(EXIT_FAILURE);
    }
    cuda_check(cudaMemcpy(dest_ptr, src_ptr, count, cudaMemcpyDeviceToHost), "cuda_copy_to_cpu");
}

__global__ void cuda_add_float_kernel(const float *a, const float *b, float *c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void cuda_add_elementwise(const float *a_host, const float *b_host, float *c_host, size_t n)
{
    // Input validation
    if (a_host == nullptr || b_host == nullptr || c_host == nullptr) {
        printf("Error: null pointer passed to cuda_add_elementwise\n");
        return;
    }
    
    if (n == 0) {
        printf("Warning: n=0 in cuda_add_elementwise\n");
        return;
    }
    
    printf("Starting cuda_add_elementwise with n=%zu\n", n);
    
    // Check if CUDA is available
    if (!cuda_available()) {
        printf("Error: CUDA not available\n");
        return;
    }
    
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    
    size_t bytes = n * sizeof(float);
    printf("Allocating %zu bytes per array\n", bytes);
    
    // Allocate device memory
    cuda_check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    cuda_check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    cuda_check(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");
    
    printf("Device memory allocated successfully\n");
    
    // Copy data to device
    cuda_check(cudaMemcpy(d_a, a_host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    cuda_check(cudaMemcpy(d_b, b_host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_b");
    
    printf("Data copied to device\n");
    
    // Launch kernel
    const int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    printf("Launching kernel with grid_size=%d, block_size=%d\n", grid_size, block_size);
    
    cuda_add_float_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    
    // Check for kernel launch errors
    cudaError_t kernel_err = cudaGetLastError();
    cuda_check(kernel_err, "cuda kernel launch");
    
    // Synchronize
    cuda_check(cudaDeviceSynchronize(), "cuda kernel sync");
    
    printf("Kernel executed successfully\n");
    
    // Copy result back
    cuda_check(cudaMemcpy(c_host, d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy c_host");
    
    printf("Result copied back to host\n");
    
    // Free device memory
    cuda_check(cudaFree(d_a), "cudaFree d_a");
    cuda_check(cudaFree(d_b), "cudaFree d_b");
    cuda_check(cudaFree(d_c), "cudaFree d_c");
    
    printf("Device memory freed\n");
}
}