#pragma once

#define NUMBLOCKS 1024 // Number of blocks for CUDA kernel execution
#define THREADSPERBLOCK 256 // Number of threads per block for CUDA kernel execution

#include "tensrCUDA.cuh"
#include "../tensr.hpp"
#include "../tensrLens.hpp"

namespace tensrCUDA {

	template<typename T>
	class tensrCUDA {
	private:
		static void launch_kernel(char op, T* a, T* b, T* result, size_t size) {
			int threadsPerBlock = std::min(static_cast<int>(size), 256);
			int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

			switch (op) {
			case '+':
				if constexpr (std::is_same_v<T, float>) {
					add_kernel_float << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else if constexpr (std::is_same_v<T, double>) {
					add_kernel_double << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else if constexpr (std::is_same_v<T, int>) {
					add_kernel_int << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else {
					throw std::runtime_error("Unsupported type for addition");
				}
				break;
			case '-':
				if constexpr (std::is_same_v<T, float>) {
					sub_kernel_float << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else if constexpr (std::is_same_v<T, double>) {
					sub_kernel_double << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else if constexpr (std::is_same_v<T, int>) {
					sub_kernel_int << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else {
					throw std::runtime_error("Unsupported type for minus");
				}
				break;
			case '*':
				if constexpr (std::is_same_v<T, float>) {
					mul_kernel_float << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else if constexpr (std::is_same_v<T, double>) {
					mul_kernel_double << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else if constexpr (std::is_same_v<T, int>) {
					mul_kernel_int << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else {
					throw std::runtime_error("Unsupported type for mult");
				}
				break;
			case '/':
				if constexpr (std::is_same_v<T, float>) {
					div_kernel_float << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else if constexpr (std::is_same_v<T, double>) {
					div_kernel_double << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else if constexpr (std::is_same_v<T, int>) {
					div_kernel_int << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else {
					throw std::runtime_error("Unsupported type for divide");
				}
				break;
			case '%':
				if constexpr (std::is_same_v<T, int>) {
					mod_kernel_int << <numBlocks, threadsPerBlock >> > (a, b, result, size);
				}
				else {
					throw std::runtime_error("Unsupported type for modulo operation");
				}
				break;
				// other operations...
			}

			cudaDeviceSynchronize();
			check_cuda_error();
		}

		static void check_cuda_error() {
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
			}
		}
	public:

		static tensr::Tensr<T> add(const tensr::Tensr<T>& a, const tensr::Tensr<T>& b) {
			return compute('+', a, b);
		}
		static tensr::Tensr<T> subtract(const tensr::Tensr<T>& a, const tensr::Tensr<T>& b) {
			return compute('-', a, b);
		}
		static tensr::Tensr<T> multiply(const tensr::Tensr<T>& a, const tensr::Tensr<T>& b) {
			return compute('*', a, b);
		}

		static tensr::Tensr<T> compute(char op, const tensr<T>& a, const tensr<T>& b) {
			// Size check
			if (a.size() != b.size()) {
				throw std::runtime_error("Tensor sizes must match");
			}

			size_t size = a.size();
			tensr<T> result(a.shape()); // Same shape as input

			// GPU memory allocation
			T* d_a = nullptr;
			T* d_b = nullptr;
			T* d_result = nullptr;

			cudaMalloc(&d_a, size * sizeof(T));
			cudaMalloc(&d_b, size * sizeof(T));
			cudaMalloc(&d_result, size * sizeof(T));

			// Copy to GPU
			cudaMemcpy(d_a, a.data(), size * sizeof(T), cudaMemcpyHostToDevice);
			cudaMemcpy(d_b, b.data(), size * sizeof(T), cudaMemcpyHostToDevice);

			// Launch kernel
			launch_kernel(op, d_a, d_b, d_result, size);

			// Copy result back
			cudaMemcpy(result.data(), d_result, size * sizeof(T), cudaMemcpyDeviceToHost);

			// Cleanup
			cudaFree(d_a);
			cudaFree(d_b);
			cudaFree(d_result);

			return result;
		}
	};

}
