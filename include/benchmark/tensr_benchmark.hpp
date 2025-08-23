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

#if defined(_WIN32) || defined(_WIN64)
#define OS_WINDOWS
#elif defined(__linux__)
#define OS_LINUX
#elif defined(__APPLE__) || defined(__MACH__)
#define OS_MAC
#endif

#if defined(__GNUC__) || defined(__clang__)
#define HAS_BUILTIN_CPU_SUPPORT 1
#else
#define HAS_BUILTIN_CPU_SUPPORT 0
#endif


#ifdef OS_LINUX
class smart_benchmark_linux
{
};
#endif
