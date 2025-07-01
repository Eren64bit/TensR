#include "tensrKernel.cuh"

__global__ void add_kernel_float(const float* a, const float* b, float* c, size_t size) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}

__global__ void add_kernel_int(const int* a, const int* b, int* c, size_t size) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}