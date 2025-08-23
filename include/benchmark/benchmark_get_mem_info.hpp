#pragma once
#include "benchmark_utils.hpp"

class get_mem_info
{
public:
    get_mem_info(benchmark_res &results) : results_(results) {}
    void detect_memory_info()
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line))
        {
            if (line.find("MemTotal") != std::string::npos)
            {
                std::regex pattern(R"(\d+)");
                std::smatch match;
                if (std::regex_search(line, match, pattern))
                {
                    size_t memory_kb = std::stoul(match[0]);
                    results_.hardware.total_memory_gb = memory_kb / (1024 * 1024);
                }
            }
            else if (line.find("MemFree") != std::string::npos)
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

private:
    benchmark_res &results_;
};
