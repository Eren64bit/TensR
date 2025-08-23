#pragma once
#include "benchmark_utils.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define HAS_BUILTIN_CPU_SUPPORT 1
#else
#define HAS_BUILTIN_CPU_SUPPORT 0
#endif

class get_cpu_info
{
public:
    get_cpu_info(benchmark_res &results) : results_(results) {}
    void detect_cpu_info()
    {
        std::string s1 = "/proc/cpuinfo";
        std::string s2 = "model name";
        results_.hardware.cpu_name = utils_benchmark::get_ifstream_info(s1, s2);
        results_.hardware.cpu_cores = std::thread::hardware_concurrency();
        std::string cpu_cmd = "lscpu | grep Architecture";
        utils_benchmark::exec_cmd_and_get_string(cpu_cmd);

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

private:
    benchmark_res results_;
};


