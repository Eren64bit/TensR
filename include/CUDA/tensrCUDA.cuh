#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Forward declarations of CUDA kernel functions for addition operations
// These functions will be defined in the corresponding .cu file.
// Perform + operations on arrays of different data types.
__global__ void add_kernel_int(const int* a, const int* b, int* c, size_t n);
__global__ void add_kernel_float(const float* a, const float* b, float* c, size_t n);
__global__ void add_kernel_double(double* a, double* b, double* c, size_t n);
//Perform - operations on arrays of different data types.
__global__ void sub_kernel_int(const int* a, const int* b, int* c, size_t n);
__global__ void sub_kernel_float(const float* a, const float* b, float* c, size_t n);
__global__ void sub_kernel_double(double* a, double* b, double* c, size_t n);
// Perform * operations on arrays of different data types.
__global__ void mul_kernel_int(const int* a, const int* b, int* c, size_t n);
__global__ void mul_kernel_float(const float* a, const float* b, float* c, size_t n);
__global__ void mul_kernel_double(double* a, double* b, double* c, size_t n);
// Perform / operations on arrays of different data types.
__global__ void div_kernel_int(const int* a, const int* b, int* c, size_t n);
__global__ void div_kernel_float(const float* a, const float* b, float* c, size_t n);
__global__ void div_kernel_double(double* a, double* b, double* c, size_t n);
// Perform % operations on arrays of different data types.
__global__ void mod_kernel_int(const int* a, const int* b, int* c, size_t n);
