#pragma once
#include "benchmark_utils.hpp"

class detect_performance
{
    public:
    detect_performance(benchmark_res &results, cuda_loader &cuda_loader) : results_(results), cuda_loader_(cuda_loader) {}

    void run_perfm_benchmark()
    {
        std::cout << "Running performance benchmarks...\n";
        detect_cpu_performance();
        detect_gpu_performance();
        std::cout << "performance benchmarks finished\n";
    }

    void detect_cpu_performance()
    {
        std::cout << "Running CPU test...\n";

        size_t small_size = 100 * 100;
        size_t medium_size = 1000 * 1000;
        size_t large_size = 2000 * 2000;

        double alloc_time_small = benchmark_memory_allocation(small_size, backend_::CPU);
        double elemwise_time_small = benchmark_elementwise_ops(small_size, backend_::CPU);
        results_.performance.small_tensors_cpu["memory_allocation"].cpu_time_ms = alloc_time_small;
        results_.performance.small_tensors_cpu["elementwise_add"].cpu_time_ms = elemwise_time_small;

        double alloc_time_medium = benchmark_memory_allocation(medium_size, backend_::CPU);
        double elemwise_time_medium = benchmark_elementwise_ops(medium_size, backend_::CPU);
        results_.performance.medium_tensors_cpu["memory_allocation"].cpu_time_ms = alloc_time_medium;
        results_.performance.medium_tensors_cpu["elementwise_add"].cpu_time_ms = elemwise_time_medium;

        double alloc_time_large = benchmark_memory_allocation(large_size, backend_::CPU);
        double elemwise_time_large = benchmark_elementwise_ops(large_size, backend_::CPU);
        results_.performance.large_tensors_cpu["memory_allocation"].cpu_time_ms = alloc_time_large;
        results_.performance.large_tensors_cpu["elementwise_add"].cpu_time_ms = elemwise_time_large;
    }

    void detect_gpu_performance() 
    {
        std::cout << "Running GPU test...\n";

        size_t small_size = 10 * 10;
        size_t medium_size = 100 * 100;
        size_t large_size = 512 * 512;

        std::cout << "CUDA available: " << (results_.capabilities.cuda_available ? "Yes" : "No") << "\n";
        std::cout << "Running CUDA alloc benchmark...\n";
        double alloc_time_small = benchmark_memory_allocation(small_size, backend_::GPU_CUDA);
        double elemwise_time_small = benchmark_elementwise_ops(small_size, backend_::GPU_CUDA);
        results_.performance.small_tensors_gpu["memory_allocation"].gpu_time_ms = alloc_time_small;
        results_.performance.small_tensors_gpu["elementwise_add"].gpu_time_ms = elemwise_time_small;

        std::cout << "Running CUDA elementwise benchmark...\n";
        double alloc_time_medium = benchmark_memory_allocation(medium_size, backend_::GPU_CUDA);
        double elemenwise_time_medium = benchmark_elementwise_ops(medium_size, backend_::GPU_CUDA);
        results_.performance.medium_tensors_gpu["memory_allocation"].gpu_time_ms = alloc_time_small;
        results_.performance.medium_tensors_gpu["elementwise_add"].gpu_time_ms = elemenwise_time_medium;

        double alloc_time_large = benchmark_memory_allocation(large_size, backend_::GPU_CUDA);
        double elemenwise_time_large = benchmark_elementwise_ops(large_size, backend_::GPU_CUDA);
        results_.performance.large_tensors_gpu["memory_allocation"].gpu_time_ms = alloc_time_large;
        results_.performance.large_tensors_gpu["elementwise_add"].gpu_time_ms = elemenwise_time_large;
    }

private:
    benchmark_res &results_;
    cuda_loader &cuda_loader_;

    double benchmark_memory_allocation(size_t elements, backend_ backend = backend_::CPU)
    {
        const int iterations = 5;
        if (results_.capabilities.cuda_available && backend == backend_::GPU_CUDA) // CUDA
        {    
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; i++)
            {
                auto& funcs = cuda_loader_.get_functions();
                if (!funcs.cuda_malloc || !funcs.cuda_free) {
                    std::cerr << "CUDA functions could not be loaded!\n";
                    return -1.0;
                }
                void *d_ptr = cuda_loader_.get_functions().cuda_malloc(elements * sizeof(float));
                if (!d_ptr)
                {
                    std::cerr << "CUDA memory allocation failed!\n";
                    return -1.0;
                }
                cuda_loader_.get_functions().cuda_free(d_ptr);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            return duration.count() / 1000.0 / iterations;
        }
        else if (backend == backend_::CPU) // CPU
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; i++)
            {
                auto ptr = std::make_unique<float[]>(elements);

                // touch memory to ensure allocation
                ptr[0] = 1.0f;
                ptr[elements - 1] = 1.0f;
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            return duration.count() / 1000.0 / iterations;
        }
        return -1.0;
    }

    double benchmark_elementwise_ops(size_t elements, backend_ backend = backend_::CPU)
    {
        const int iterations = 5;
        if (results_.capabilities.cuda_available && backend == backend_::GPU_CUDA) // CUDA
        {    
            auto a_ = std::make_unique<float[]>(elements);
            auto b_ = std::make_unique<float[]>(elements);
            auto c_ = std::make_unique<float[]>(elements);
            // Initialize
            for (size_t i = 0; i < elements; ++i)
            {
                a_[i] = static_cast<float>(i);
                b_[i] = static_cast<float>(i) * 2.0f;
            }
            
            cuda_loader_.get_functions().cuda_sync(); // Ensure any previous operations are complete
            auto start = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < iterations; ++iter)
            {
                cuda_loader_.get_functions().cuda_add_elementwise(a_.get(), b_.get(), c_.get(), elements);
            }
            cuda_loader_.get_functions().cuda_sync(); // Wait for all operations to finish
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            return duration.count() / 1000.0 / iterations; // convert to ms
        }
        else if (backend == backend_::CPU) // CPU
        {
            auto a_ = std::make_unique<float[]>(elements);
            auto b_ = std::make_unique<float[]>(elements);
            auto c_ = std::make_unique<float[]>(elements);

            // Initialize
            for (size_t i = 0; i < elements; ++i)
            {
                a_[i] = static_cast<float>(i);
                b_[i] = static_cast<float>(i) * 2.0f;
            }

            auto start = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < elements; ++i)
                {
                    c_[i] = a_[i] + b_[i];
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            return duration.count() / 1000.0 / iterations; // convert to ms
        }
        return -1.0;
    }
};
