#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <thread>
#include <array>
#include <filesystem>
#include <map>
#include <vector>
#include <regex>
#include <immintrin.h>
#include <dlfcn.h>
#include <chrono>

enum class backend_
{
    CPU,
    GPU_CUDA,
    GPU_AMD,
    GPU_OTHERS
};

// ------------- Benchmark result -------------
struct benchmark_res
{
    struct Hardware
    {
        std::string cpu_name;
        int cpu_cores;
        std::string cpu_architecture;
        std::vector<std::string> gpu_names;
        size_t available_gpu_memory_gb;
        size_t total_memory_gb;
        size_t free_memory_gb;
    } hardware;

    struct Capabilities
    {
        bool sse_support = false;
        bool avx_support = false;
        bool avx2_support = false;
        bool avx512_support = false;
        bool cuda_available = false;
        bool opencl_available = false;
        std::string cuda_version;
        std::vector<std::string> cuda_devices;
    } capabilities;

    struct Performance
    {
        struct op_res
        {
            double cpu_time_ms;
            double gpu_time_ms; // -1 if not available
            double memory_bandwidth_gb_s;
        };

        std::map<std::string, op_res> small_tensors_cpu;  // 100x100
        std::map<std::string, op_res> medium_tensors_cpu; // 1000x1000
        std::map<std::string, op_res> large_tensors_cpu;  // 10000x10000

        std::map<std::string, op_res> small_tensors_gpu;  // 100x100
        std::map<std::string, op_res> medium_tensors_gpu; // 1000x1000
        std::map<std::string, op_res> large_tensors_gpu;  // 10000x10000
    } performance;
};

struct cuda_functions
{
    // FIX: cuda_available returns bool, not int
    using cuda_available_t = bool (*)(); // Was: int (*)()
    using cuda_malloc_t = void *(*)(size_t);
    using cuda_free_t = void (*)(void *);
    using cuda_sync_t = void (*)();
    using cuda_copy_to_cpu_t = void (*)(void *, const void *, size_t);
    using cuda_copy_to_gpu_t = void (*)(void *, const void *, size_t);
    using cuda_add_elementwise_t = void (*)(const float *, const float *, float *, size_t);

    cuda_available_t cuda_available = nullptr;
    cuda_malloc_t cuda_malloc = nullptr;
    cuda_free_t cuda_free = nullptr;
    cuda_sync_t cuda_sync = nullptr;
    cuda_copy_to_cpu_t cuda_copy_to_cpu = nullptr;
    cuda_copy_to_gpu_t cuda_copy_to_gpu = nullptr;
    cuda_add_elementwise_t cuda_add_elementwise = nullptr;
};

class cuda_loader
{
private:
    void *handle_ = nullptr;
    cuda_functions funcs_;

public:
    explicit cuda_loader(const char *lib_path)
    {
        handle_ = dlopen(lib_path, RTLD_LAZY);
        if (!handle_)
        {
            const char *error = dlerror();
            throw std::runtime_error(std::string("CUDA lib not found: ") + (error ? error : "Unknown error"));
        }

        // Clear any existing errors
        dlerror();

        // Load functions with error checking
        funcs_.cuda_available = (cuda_functions::cuda_available_t)dlsym(handle_, "cuda_available");
        check_dlsym_error("cuda_available");

        funcs_.cuda_malloc = (cuda_functions::cuda_malloc_t)dlsym(handle_, "cuda_malloc");
        check_dlsym_error("cuda_malloc");

        funcs_.cuda_free = (cuda_functions::cuda_free_t)dlsym(handle_, "cuda_free");
        check_dlsym_error("cuda_free");

        funcs_.cuda_sync = (cuda_functions::cuda_sync_t)dlsym(handle_, "cuda_sync");
        check_dlsym_error("cuda_sync");

        funcs_.cuda_copy_to_cpu = (cuda_functions::cuda_copy_to_cpu_t)dlsym(handle_, "cuda_copy_to_cpu");
        check_dlsym_error("cuda_copy_to_cpu");

        funcs_.cuda_copy_to_gpu = (cuda_functions::cuda_copy_to_gpu_t)dlsym(handle_, "cuda_copy_to_gpu");
        check_dlsym_error("cuda_copy_to_gpu");

        funcs_.cuda_add_elementwise = (cuda_functions::cuda_add_elementwise_t)dlsym(handle_, "cuda_add_elementwise");
        check_dlsym_error("cuda_add_elementwise");

        printf("CUDA library loaded successfully\n");
        printf("Function pointers:\n");
        printf("  cuda_available: %p\n", (void *)funcs_.cuda_available);
        printf("  cuda_malloc: %p\n", (void *)funcs_.cuda_malloc);
        printf("  cuda_free: %p\n", (void *)funcs_.cuda_free);
    }

    const cuda_functions &get_functions() const { return funcs_; }

    bool is_loaded() const { return handle_ != nullptr; }

    ~cuda_loader()
    {
        if (handle_)
        {
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

private:
    void check_dlsym_error(const char *symbol_name)
    {
        const char *error = dlerror();
        if (error)
        {
            dlclose(handle_);
            handle_ = nullptr;
            throw std::runtime_error(std::string("Failed to load symbol ") + symbol_name + ": " + error);
        }
    }

    // Prevent copying
    cuda_loader(const cuda_loader &) = delete;
    cuda_loader &operator=(const cuda_loader &) = delete;
};

// Additional debug function to test the loader
void test_cuda_loader(const cuda_loader &loader)
{
    printf("=== Testing CUDA Loader ===\n");

    const auto &funcs = loader.get_functions();

    // Test cuda_available first
    if (funcs.cuda_available)
    {
        printf("Testing cuda_available...\n");
        bool available = funcs.cuda_available();
        printf("CUDA available: %s\n", available ? "Yes" : "No");

        if (!available)
        {
            printf("CUDA not available, skipping memory tests\n");
            return;
        }
    }
    else
    {
        printf("cuda_available function not loaded!\n");
        return;
    }

    // Test memory allocation
    if (funcs.cuda_malloc && funcs.cuda_free)
    {
        printf("Testing memory allocation...\n");

        const size_t test_size = 1024; // 1KB test
        void *test_ptr = funcs.cuda_malloc(test_size);

        if (test_ptr)
        {
            printf("✓ Allocated %zu bytes at %p\n", test_size, test_ptr);
            funcs.cuda_free(test_ptr);
            printf("✓ Memory freed successfully\n");
        }
        else
        {
            printf("✗ Memory allocation failed\n");
        }
    }
    else
    {
        printf("Memory functions not loaded!\n");
    }

    printf("=== Loader Test Complete ===\n");
}

namespace utils_benchmark
{
    // helper functions
    std::string exec_command(const std::string &cmd)
    {
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);

        if (!pipe)
        {
            return "";
        }

        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
        {
            result += buffer.data();
        }

        return result;
    }

    std::string trim(const std::string &str)
    {
        size_t first = str.find_first_not_of(" \t\n\r");
        size_t last = str.find_last_not_of(" \t\n\r");
        return (first == std::string::npos || last == std::string::npos) ? "" : str.substr(first, last - first + 1);
    }

    std::string exec_cmd_and_get_string(const std::string &cmd)
    {
        std::string result = exec_command(cmd);
        return trim(result);
    }

    std::string get_ifstream_info(const std::string &path, const std::string &key)
    {
        std::ifstream file(path);
        if (!file.is_open())
            return "";
        std::string content;
        while (std::getline(file, content))
        {
            if (content.find(key) != std::string::npos)
            {
                size_t pos = content.find(":");
                if (pos != std::string::npos)
                {
                    return trim(content.substr(pos + 1));
                }
                else
                {
                    return "";
                }
            }
        }
        return ""; // <-- eksik return eklendi
    }

    std::string get_library_dir()
    {
        Dl_info info;
        dladdr((void *)get_library_dir, &info);
        return std::filesystem::path(info.dli_fname).parent_path().string();
    }

    static std::string default_output_path()
    {
        const char *home = std::getenv("HOME");
        std::string base = home ? std::string(home) : std::string("/tmp");
        std::string dir = base + "/.stargazer";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        return dir + "/benchmark_conf.txt";
    }

}