#pragma once
#include "benchmark_get_cpu_info.hpp"
#include "benchmark_get_gpu_info.hpp"
#include "benchmark_get_mem_info.hpp"
#include "benchmark_performance.hpp"
#include "benchmark_write_read_file.hpp"

#if defined(_WIN32) || defined(_WIN64)
#define OS_WINDOWS
#elif defined(__linux__)
#define OS_LINUX
#elif defined(__APPLE__) || defined(__MACH__)
#define OS_MAC
#endif

#ifdef OS_LINUX
class smart_benchmark_linux
{
public:
    smart_benchmark_linux() = default;
    ~smart_benchmark_linux() = default;

    void run_all_benchmarks()
    {
        std::cout << "Running benchmarks on Linux...\n";
        benchmark_res results_;
        std::string lib_path = utils_benchmark::get_library_dir() + "/libcuda_tensr_api.so";
        cuda_loader cuda_loader_(lib_path.c_str());
        get_cpu_info cpu_info(results_);
        cpu_info.detect_cpu_info();
        get_mem_info mem_info(results_);
        mem_info.detect_memory_info();
        get_gpu_info gpu_info(results_);
        gpu_info.detect_gpu_info();
        detect_performance performance(results_, cuda_loader_);
        performance.run_perfm_benchmark();
        write_read_file file_io(results_);
        std::string output_path = utils_benchmark::default_output_path();
        file_io.save_to_file(output_path);
        std::cout << "Benchmark results saved to " << output_path << "\n";
        
    }
};
#endif
