cmake_minimum_required(VERSION 2.6)
project(cs267_hw1)

# BLAS
find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIR})

## C++ files
# Build O3 optimized binaries - C++
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic -Wall -std=gnu99")

# Naive
add_executable(benchmark-naive benchmark.c dgemm-naive.c)
target_link_libraries(benchmark-naive blas rt)

add_executable(benchmark-blocked benchmark.c dgemm-blocked.c)
target_link_libraries(benchmark-blocked blas rt)

add_executable(benchmark-blas benchmark.c dgemm-blas.c)
target_link_libraries(benchmark-blas blas rt)

