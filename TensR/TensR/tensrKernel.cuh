#include <cuda_runtime.h>
#include <stddef.h>
#include <device_launch_parameters.h>

__global__ void add_kernel_float(const float* a, const float* b, float* c, size_t size);

__global__ void add_kernel_int(const int* a, const int* b, int* c, size_t size);