#include <iostream>
#include "include/tensr_static.hpp"
#include "include/benchmark/tensr_benchmark.hpp"

int main() {

    smart_benchmark_linux bench;
    bench.run_all_benchmarks();
    return 0;
}