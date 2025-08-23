#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

    bool cuda_available();
    void cuda_check(int result, const char *msg = "CUDA error");
    void *cuda_malloc(size_t size);
    void cuda_free(void *ptr);
    void cuda_sync();
    void cuda_copy_to_gpu(void *dest_ptr, const void *src_ptr, size_t count);
    void cuda_copy_to_cpu(void *dest_ptr, const void *src_ptr, size_t count);

#ifdef __cplusplus
}
#endif