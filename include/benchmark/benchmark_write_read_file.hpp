#pragma once

#include "benchmark_utils.hpp"

class write_read_file
{
public:
    write_read_file(benchmark_res &results) : results_(results) {}
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

        file << "  \"performance_cpu\": {\n";
        file << "    \"small_tensors_cpu\": " << format_performance_results(results_.performance.small_tensors_cpu) << ",\n";
        file << "    \"medium_tensors_cpu\": " << format_performance_results(results_.performance.medium_tensors_cpu) << ",\n";
        file << "    \"large_tensors_cpu\": " << format_performance_results(results_.performance.large_tensors_cpu) << "\n";
        file << "  }\n";
        file << "}";

        file << "  \"performance_gpu\": {\n";
        file << "    \"small_tensors_gpu\": " << format_performance_results(results_.performance.small_tensors_gpu) << ",\n";
        file << "    \"medium_tensors_gpu\": " << format_performance_results(results_.performance.medium_tensors_gpu) << ",\n";
        file << "    \"large_tensors_gpu\": " << format_performance_results(results_.performance.large_tensors_gpu) << "\n";
        file << "  }\n";
        file << "}";

        return file.str();
    }

    void save_to_file(const std::string &filename)
    {
        std::ofstream ofs(filename);
        if (!ofs.is_open())
        {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return;
        }

        ofs << generate_file();
        ofs.close();
    }

    std::string format_performance_results(const std::map<std::string, benchmark_res::Performance::op_res> &results)
    {
        std::ostringstream json;
        json << "{\n";

        bool first = true;
        for (const auto &[op, result] : results)
        {
            if (!first)
                json << ",\n";
            json << "      \"" << op << "\": {\n";
            json << "        \"cpu_time_ms\": " << result.cpu_time_ms << ",\n";
            json << "        \"gpu_time_ms\": " << result.gpu_time_ms << ",\n";
            json << "        \"memory_bandwidth_gb_s\": " << result.memory_bandwidth_gb_s << "\n";
            json << "      }";
            first = false;
        }

        json << "\n    }";
        return json.str();
    }

private:
benchmark_res & results_;
};
