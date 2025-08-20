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

#if defined(_WIN32) || defined(_WIN64)
#define OS_WINDOWS
#elif defined(__linux__)
#define OS_LINUX
#elif defined(__APPLE__) || defined(__MACH__)
#define OS_MAC
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define HAS_BUILTIN_CPU_SUPPORT 1
#else
#define HAS_BUILTIN_CPU_SUPPORT 0
#endif

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

static std::string default_output_path()
{
    const char *home = std::getenv("HOME");
    std::string base = home ? std::string(home) : std::string("/tmp");
    std::string dir = base + "/.stargaze";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir + "/benchmark_config.txt";
}

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

        std::map<std::string, op_res> small_tensors;  // 100x100
        std::map<std::string, op_res> medium_tensors; // 1000x1000
        std::map<std::string, op_res> large_tensors;  // 5000x5000
    } performance;
};

#ifdef OS_LINUX
class smart_benchmark_linux
{
private:
    benchmark_res results_;

public:
    void run_full_benchmark()
    {
        std::cout << "Starting Linux system benchmark...\n";
        detect_hardware();
        detect_capabilities();
        save_to_file();
        std::cout << "Benchmark completed!\n";
    }

    void save_to_file(std::string path = default_output_path())
    {
        std::cout << "Starting writing to file..." << "\n";
        std::ofstream file(path);
        file << generate_file();
        file.close();
        std::cout << "Finished writing to file" << "\n";
        std::cout << "Result saved to benchmark_config.txt" << "\n";
    }

private:
    // ------------- Detect Hardware -------------
    void detect_hardware()
    {
        std::cout << "Detecting hardware...\n";
        detect_cpu_info();
        detect_gpu_info();
        detect_memory_info();
        std::cout << "Detecting hardware completed!\n";
    }

    void detect_cpu_info()
    {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;

        while (std::getline(cpuinfo, line))
        {
            if (line.find("model name") != std::string::npos)
            {
                size_t pos = line.find(":");
                if (pos != std::string::npos)
                {
                    results_.hardware.cpu_name = line.substr(pos + 2);
                }
            }
        }

        //  Count cpu cores
        results_.hardware.cpu_cores = std::thread::hardware_concurrency();

        // Get cpu Architecture
        std::string cpu_cmd = "lscpu | grep Architecture";
        std::string res = exec_command(cpu_cmd);
        std::stringstream ss(res);

        std::string cpu_arch;
        while (std::getline(ss, cpu_arch))
        {
            if (!cpu_arch.empty())
            {
                cpu_arch.erase(0, cpu_arch.find_first_not_of(" \t"));
                cpu_arch.erase(cpu_arch.find_last_not_of(" \t"));
                results_.hardware.cpu_architecture = cpu_arch;
            }
        }
    }

    void detect_gpu_info()
    {
        // Nvidia
        if (system("nvidia-smi > /dev/null 2>&1") == 0)
        {
            std::string cmd = "nvidia-smi --query-gpu=name --format=csv,noheader,nounits";
            std::string result = exec_command(cmd);

            std::stringstream ss(result);
            std::string gpu_name;
            while (std::getline(ss, gpu_name))
            {
                gpu_name.erase(0, gpu_name.find_first_not_of(" \t"));
                gpu_name.erase(gpu_name.find_last_not_of(" \t"));
                results_.hardware.gpu_names.push_back(gpu_name);
            }

            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits";
            result = exec_command(cmd);
            if (!result.empty())
            {
                results_.hardware.available_gpu_memory_gb = std::stoi(result) / 1024;
            }
        }

        // Amd
        if (std::filesystem::exists("/sys/class/drm"))
        {
            for (const auto &entry : std::filesystem::directory_iterator("/sys/class/drm"))
            {
                std::string path = entry.path().string() + "/device/uevent";
                if (std::filesystem::exists(path))
                {
                    std::ifstream file(path);
                    std::string line;
                    while (std::getline(file, line))
                    {
                        if (line.find("PCI_SUBSYS_NAME=") != std::string::npos)
                        {
                            size_t pos = line.find("=");
                            if (pos != std::string::npos)
                            {
                                std::string gpu_name = line.substr(pos + 1);
                                if (gpu_name.find("AMD") != std::string::npos ||
                                    gpu_name.find("Radeon") != std::string::npos)
                                {
                                    results_.hardware.gpu_names.push_back(gpu_name);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void detect_memory_info()
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;

        while (std::getline(meminfo, line))
        {
            if (line.find("MemTotal:") != std::string::npos)
            {
                std::regex pattern(R"(\d+)");
                std::smatch match;
                if (std::regex_search(line, match, pattern))
                {
                    size_t memory_kb = std::stoul(match[0]);
                    results_.hardware.total_memory_gb = memory_kb / (1024 * 1024);
                }
            }
            else if (line.find("MemFree:") != std::string::npos)
            {
                std::regex pattern(R"(\d+)");
                std::smatch match;
                if (std::regex_search(line, match, pattern))
                {
                    size_t memory_kb = std::stoul(match[0]);
                    results_.hardware.total_memory_gb = memory_kb / (1024 * 1024);
                }
            }
        }
    }
    // ------------- Detect Hardware -------------

    // ------------- Detect Capabilities -------------
    void detect_capabilities()
    {
        std::cout << "Detecting Capabilities...\n";
        detect_simd_support();
        detect_cuda_support();
        detect_opencl_support();
    }

    void detect_simd_support()
    {
#if HAS_BUILTIN_CPU_SUPPORT
        results_.capabilities.sse_support = __builtin_cpu_supports("sse");
        results_.capabilities.avx_support = __builtin_cpu_supports("avx");
        results_.capabilities.avx2_support = __builtin_cpu_supports("avx2");
        results_.capabilities.avx512_support = __builtin_cpu_supports("avx512f");
#else
        std::string res = exec_command("lscpu | grep Flags");
        results_.capabilities.sse_support = (res.find("sse") != std::string::npos);
        results_.capabilities.avx_support = (res.find("avx") != std::string::npos);
        results_.capabilities.avx2_support = (res.find("avx2") != std::string::npos);
        results_.capabilities.avx512_support = (res.find("avx512f") != std::string::npos);
#endif
    }

    void detect_cuda_support()
    {
        if (system("nvcc --version > /dev/null 2>&1") == 0)
        {
            results_.capabilities.cuda_available = true;

            std::string result = exec_command("nvcc --version | grep 'release' | awk '{print $6}'");
            if (!result.empty())
            {
                results_.capabilities.cuda_version = result;
                results_.capabilities.cuda_version.pop_back(); // Remove newline
            }

            result = exec_command("nvidia-smi --query-gpu=name --format=csv,noheader,nounits");
            std::stringstream ss(result);
            std::string device;
            while (std::getline(ss, device))
            {
                if (!device.empty())
                {
                    device.erase(0, device.find_first_not_of(" \t"));
                    device.erase(device.find_last_not_of(" \t") + 1);
                    results_.capabilities.cuda_devices.push_back(device);
                }
            }
        }
    }

    void detect_opencl_support()
    {
        results_.capabilities.opencl_available = std::filesystem::exists("/usr/lib/libOpenCL.so") ||
                                                 std::filesystem::exists("/usr/lib64/libOpenCL.so");
    }
    //  ------------- Detect Capabilities  -------------

    // ------------- Generate File -------------
    std::string generate_file()
    {
        std::ostringstream file;
        file << "{\n";
        file << "  \"hardware\": {\n";
        file << "    \"cpu_name\": \"" << results_.hardware.cpu_name << "\",\n";
        file << "    \"cpu_cores\": " << results_.hardware.cpu_cores << ",\n";
        file << "    \"cpu_architecture\": \"" << results_.hardware.cpu_architecture << "\",\n";
        file << "    \"total_memory_gb\": " << results_.hardware.total_memory_gb << ",\n";
        file << "    \"free_memory_gb\":  " << results_.hardware.free_memory_gb << ",\n";
        file << "    \"gpu_names\": [";

        for (size_t i = 0; i < results_.hardware.gpu_names.size(); ++i)
        {
            file << "\"" << results_.hardware.gpu_names[i] << "\"";
            if (i < results_.hardware.gpu_names.size() - 1)
                file << ", ";
        }

        file << "],\n";
        file << "    \"available_gpu_memory_gb\": " << results_.hardware.available_gpu_memory_gb << "\n";
        file << "  },\n";

        file << "  \"capabilities\": {\n";
        file << "    \"sse_support\": " << (results_.capabilities.sse_support ? "true" : "false") << ",\n";
        file << "    \"avx_support\": " << (results_.capabilities.avx_support ? "true" : "false") << ",\n";
        file << "    \"avx2_support\": " << (results_.capabilities.avx2_support ? "true" : "false") << ",\n";
        file << "    \"avx512_support\": " << (results_.capabilities.avx512_support ? "true" : "false") << ",\n";
        file << "    \"cuda_available\": " << (results_.capabilities.cuda_available ? "true" : "false") << ",\n";
        file << "    \"cuda_version\": \"" << results_.capabilities.cuda_version << "\",\n";
        file << "    \"opencl_available\": " << (results_.capabilities.opencl_available ? "true" : "false") << "\n";
        file << "  },\n";

        return file.str();
    }
    // ------------- Generate File -------------
};
#endif
