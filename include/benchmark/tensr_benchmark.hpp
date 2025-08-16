#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(_WIN32) || defined(_WIN64)
#define OS_WINDOWS
#elif defined(__linux__)
#define OS_LINUX
#elif defined(__APPLE__) || defined(__MACH__)
#define OS_MAC
#endif

#ifdef __CUDAC__
#include <cuda_runtime.h>
#endif



