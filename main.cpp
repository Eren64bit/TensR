#include <iostream>
#include "include/tensr_static.hpp"
#include "include/benchmark/tensr_benchmark.hpp"

int main() {

    smart_benchmark_linux bench;
    bench.run_full_benchmark();
    return 0;
}