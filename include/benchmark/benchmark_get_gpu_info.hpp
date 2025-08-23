#pragma once

#include "benchmark_utils.hpp"

class get_gpu_info
{
public:
    get_gpu_info(benchmark_res &results) : results_(results) {}
    void detect_gpu_info()
    {
        // Nvidia
        if (system("nvidia-smi > /dev/null 2>&1") == 0)
        {
            std::string cmd = "nvidia-smi --query-gpu=name --format=csv,noheader,nounits";
            std::string result = utils_benchmark::exec_command(cmd);

            std::stringstream ss(result);
            std::string gpu_name;
            while (std::getline(ss, gpu_name))
            {
                results_.hardware.gpu_names.push_back(utils_benchmark::trim(gpu_name));
            }

            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits";
            result = utils_benchmark::exec_command(cmd);
            if (!result.empty())
            {
                results_.hardware.available_gpu_memory_gb = std::stoi(result) / 1024;
            }
        }

        // CUDA
        if (system("nvcc --version > /dev/null 2>&1") == 0)
        {
            results_.capabilities.cuda_available = true;

            std::string result = utils_benchmark::exec_command("nvcc --version | grep 'release' | awk '{print $6}'");
            if (!result.empty())
            {
                results_.capabilities.cuda_version = result;
                results_.capabilities.cuda_version.pop_back(); // Remove newline
            }

            result = utils_benchmark::exec_command("nvidia-smi --query-gpu=name --format=csv,noheader,nounits");
            std::stringstream ss(result);
            std::string device;
            while (std::getline(ss, device))
            {
                if (!device.empty())
                {
                    results_.capabilities.cuda_devices.push_back(utils_benchmark::trim(device));
                }
            }
        }

        // OpenCL
        results_.capabilities.opencl_available = std::filesystem::exists("/usr/lib/libOpenCL.so") ||
                                                 std::filesystem::exists("/usr/lib64/libOpenCL.so");

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

private:
    benchmark_res &results_;
};
