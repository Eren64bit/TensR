
#include <cuda_runtime.h>

inline bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess) && (device_count > 0);
}