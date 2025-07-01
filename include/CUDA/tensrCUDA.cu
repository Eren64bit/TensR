
#include "tensrCUDA.cuh"

// Perform + operations on arrays of different data types.
__global__ void add_kernel_int(const int* a, const int* b, int* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}
__global__ void add_kernel_float(const float* a, const float* b, float* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}
__global__ void add_kernel_double(double* a, double* b, double* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}
//Perform - operations on arrays of different data types.
__global__ void sub_kernel_int(const int* a, const int* b, int* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] - b[idx];
	}
}
__global__ void sub_kernel_float(const float* a, const float* b, float* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] - b[idx];
	}
}
__global__ void sub_kernel_double(double* a, double* b, double* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] - b[idx];
	}
}
// Perform * operations on arrays of different data types.
__global__ void mul_kernel_int(const int* a, const int* b, int* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] * b[idx];
	}
}
__global__ void mul_kernel_float(const float* a, const float* b, float* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] * b[idx];
	}
}
__global__ void mul_kernel_double(double* a, double* b, double* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] * b[idx];
	}
}
// Perform / operations on arrays of different data types.
__global__ void div_kernel_int(const int* a, const int* b, int* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		if (b[idx] != 0) { // Avoid division by zero
			c[idx] = a[idx] / b[idx];
		} else {
			c[idx] = 0; // Handle division by zero case
		}
	}
}
__global__ void div_kernel_float(const float* a, const float* b, float* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		if (b[idx] != 0) { // Avoid division by zero
			c[idx] = a[idx] / b[idx];
		}
		else {
			c[idx] = 0; // Handle division by zero case
		}
	}
}
__global__ void div_kernel_double(double* a, double* b, double* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		if (b[idx] != 0) { // Avoid division by zero
			c[idx] = a[idx] / b[idx];
		}
		else {
			c[idx] = 0; // Handle division by zero case
		}
	}
}
// Perform % operations on arrays of different data types.
__global__ void mod_kernel_int(const int* a, const int* b, int* c, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		if (b[idx] != 0) { // Avoid division by zero
			c[idx] = a[idx] % b[idx];
		} else {
			c[idx] = 0; // Handle division by zero case
		}
	}
}
