cmake_minimum_required(VERSION 3.10)
project(TensrTest)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(tensr_test tests/main.cpp)