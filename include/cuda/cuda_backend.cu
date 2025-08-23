#include <tensr_cuda_api.cuh>
#include <stdio.h>

bool cuda_available()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess) && (device_count > 0);
}

void cuda_check(int result, const char *msg = "CUDA error")
{
    cudaError_t cuda_result = static_cast<cudaError_t>(result);
    if (cuda_result != cudaSuccess)
    {
        printf("CUDA Error: %s - %s\n", msg, cudaGetErrorString(cuda_result));
    } 
}

void *cuda_malloc(size_t size)
{
    void *new_ptr = nullptr;
    cuda_check(cudaMalloc(&new_ptr, size));
    return new_ptr;
}

void cuda_free(void *ptr)
{
    cudaFree(ptr);
}

void cuda_sync()
{
    cudaDeviceSynchronize();
}

void cuda_copy_to_gpu(void *dest_ptr, const void *src_ptr, size_t count)
{
    cuda_check(cudaMemcpy(dest_ptr, src_ptr, count, cudaMemcpyHostToDevice));
}

void cuda_copy_to_cpu(void *dest_ptr, const void *src_ptr, size_t count)
{
    cuda_check(cudaMemcpy(dest_ptr, src_ptr, count, cudaMemcpyDeviceToHost));
}