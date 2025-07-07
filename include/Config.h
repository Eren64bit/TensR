#pragma once

#define TENSR_VERSION_MAJOR 0
#define TENSR_VERSION_MINOR 1
#define TENSR_VERSION_PATCH 0
#define TENSR_VERSION_STRING "0.1.0"

#define TENSR_MODE_NORMAL 0
#define TENSR_MODE_LAZY 1
#define TENSR_MODE_CUDA 2

#ifndef TENSR_MODE
#define TENSR_MODE TENSR_MODE_NORMAL
#endif

#if defined(__CUDACC__) || defined(__CUDABUILD__)
    #define TENSR_HAS_CUDA 1
#else
    #define TENSR_HAS_CUDA 0
#endif

#if TENSR_MODE == TENSR_MODE_CUDA && TENSR_HAS_CUDA == 1
    #define TENSR_CUDA_ENABLED 1
#else
    #define TENSR_CUDA_ENABLED 0
#endif

#if TENSR_MODE == TENSR_MODE_LAZY
    #define TENSR_LAZY_ENABLED 1
#else
    #define TENSR_LAZY_ENABLED 0
#endif

#ifndef TENSR_DEBUG
#define TENSR_DEBUG 0
#endif

#ifndef TENSR_OPTIMIZED
#define TENSR_OPTIMIZED 1
#endif


#define TENSR_AUTHOR_E "Eren64bit"
#define TENSR_AUTHOR_A "Ali"