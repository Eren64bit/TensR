#pragma once


/*
void detect_performance()
    {
        std::cout << "Running performance benchmarks...\n";
        detect_cpu_performance();
        detect_gpu_performance();
        std::cout << "performance benchmarks finished\n";
    }

    void detect_cpu_performance()
    {
        std::cout << "Running CPU test...\n";
        benchmark_tensor_operations("small", {100, 100}, backend_::CPU);
        benchmark_tensor_operations("medium", {1000, 1000}, backend_::CPU);
        benchmark_tensor_operations("large", {10000, 10000}, backend_::CPU);
    }

    void detect_gpu_performance()
    {
        std::cout << "Running GPU test...\n";
        if (results_.capabilities.cuda_available == true)
        {
            std::string so_path = get_library_dir() + "tensr_cuda_api.cuh";
            cuda_loader loader(so_path.c_str());
            const cuda_functions &f = loader.get_functions();
            benchmark_tensor_operations("small", {100, 100}, backend_::GPU_CUDA);
            benchmark_tensor_operations("medium", {1000, 1000}, backend_::GPU_CUDA);
            benchmark_tensor_operations("large", {10000, 10000}, backend_::GPU_CUDA);
        }
    }

    void benchmark_tensor_operations(const std::string &size_label,
                                     const std::vector<size_t> &shape,
                                     backend_ backend = backend_::CPU,
                                     cuda_functions *cuda_funcs = nullptr)
    {
        std::cout << "Benchmarking " << size_label << " tensors ("
                  << shape[0] << "x" << shape[1] << ")...\n";
        size_t elements = shape[0] * shape[1];
        if (backend == backend_::CPU)
        {
            auto alloc_time = benchmark_memory_allocation(elements, backend, cuda_funcs);
            auto elementswise_time = benchmark_elementwise_ops(elements, backend, cuda_funcs);

            if (size_label == "small")
            {
                results_.performance.small_tensors_cpu["elementwise_add"].cpu_time_ms = elementswise_time;
                results_.performance.small_tensors_cpu["memory_allocation"].cpu_time_ms = alloc_time;
            }
            else if (size_label == "medium")
            {
                results_.performance.medium_tensors_cpu["elementwise_add"].cpu_time_ms = elementswise_time;
                results_.performance.medium_tensors_cpu["memory_allocation"].cpu_time_ms = alloc_time;
            }
            else if (size_label == "large")
            {
                results_.performance.large_tensors_cpu["elementwise_add"].cpu_time_ms = elementswise_time;
                results_.performance.large_tensors_cpu["memory_allocation"].cpu_time_ms = alloc_time;
            }
        }
        else if (backend == backend_::GPU_CUDA && cuda_funcs != nullptr)
        {
            auto alloc_time = benchmark_memory_allocation(elements, backend, cuda_funcs);
            //auto elementswise_time = benchmark_elementwise_ops(elements, backend, cuda_funcs);

            if (size_label == "small")
            {
                results_.performance.small_tensors_gpu["memory_allocation"].gpu_time_ms = alloc_time;
                //results_.performance.small_tensors_gpu["elementwise_add"].gpu_time_ms = elementswise_time;
            }
            else if (size_label == "medium")
            {
                results_.performance.medium_tensors_gpu["memory_allocation"].gpu_time_ms = alloc_time;
                //results_.performance.medium_tensors_gpu["elementwise_add"].gpu_time_ms = elementswise_time;
            }
            else if (size_label == "large")
            {
                results_.performance.large_tensors_gpu["memory_allocation"].gpu_time_ms = alloc_time;
                //results_.performance.large_tensors_gpu["elementwise_add"].gpu_time_ms = elementswise_time;
            }
        }
    }

    double benchmark_memory_allocation(size_t elements, backend_ backend = backend_::CPU, cuda_functions *cuda_funcs = nullptr)
    {
        const int iterations = 100;
        if (backend == backend_::CPU) // CPU
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
        else if (backend == backend_::GPU_CUDA && cuda_funcs != nullptr) // CUDA
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; i++)
            {
                void *d_ptr = cuda_funcs->cuda_malloc(elements * sizeof(float));
                cuda_funcs->cuda_free(d_ptr);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            return duration.count() / 1000.0 / iterations;
        }
        else
        {
            return -1.0;
        }
    }

    double benchmark_elementwise_ops(size_t elements, backend_ backend = backend_::CPU, cuda_functions *cuda_funcs = nullptr)
    {
        auto a_ = std::make_unique<float[]>(elements);
        auto b_ = std::make_unique<float[]>(elements);
        auto c_ = std::make_unique<float[]>(elements);

        if (backend == backend_::CPU)
        {
            // Initialize
            for (size_t i = 0; i < elements; ++i)
            {
                a_[i] = static_cast<float>(i);
                b_[i] = static_cast<float>(i) * 2.0f;
            }

            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < elements; ++i)
            {
                c_[i] = a_[i] + b_[i];
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            return duration.count() / 1000.0; // convert to ms
        }
        else if (backend == backend_::GPU_CUDA && cuda_funcs != nullptr)
        {
        }
        else
        {
            return -1.0;
        }
    }

    double benchmark_matrix_multiplication(size_t N, backend_ backend = backend_::CPU, cuda_functions *cuda_funcs = nullptr)
    {
        auto a_ = std::make_unique<float[]>(N * N);
        auto b_ = std::make_unique<float[]>(N * N);
        auto c_ = std::make_unique<float[]>(N * N);

        if (backend == backend_::CPU)
        {
            // Initialize
            for (size_t i = 0; i < N * N; ++i)
            {
                a_[i] = static_cast<float>(i);
                b_[i] = static_cast<float>(i) * 2.0f;
            }

            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < N; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    c_[i * N + j] = 0.0f;
                    for (size_t k = 0; k < N; ++k)
                    {
                        c_[i * N + j] += a_[i * N + k] * b_[k * N + j];
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            return duration.count() / 1000.0; // convert to ms
        }
        else if (backend == backend_::GPU_CUDA && cuda_funcs != nullptr)
        {
        }
        else
        {
            return -1.0;
        }
    }

    */