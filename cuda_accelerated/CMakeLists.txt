cmake_minimum_required(VERSION 3.18)
project(HFT_CUDA_Accelerated LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures (Turing, Ampere, Ada, Hopper)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/host
    ${CUDA_INCLUDE_DIRS}
)

# ============================================================================
# Library 1: Monte Carlo CUDA Kernels
# ============================================================================
add_library(monte_carlo_cuda STATIC
    kernels/monte_carlo_kernel.cu
)

target_link_libraries(monte_carlo_cuda
    CUDA::cudart
    CUDA::curand
)

set_target_properties(monte_carlo_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Library 2: Backtesting CUDA Kernels
# ============================================================================
add_library(backtest_cuda STATIC
    kernels/backtest_kernel.cu
)

target_link_libraries(backtest_cuda
    CUDA::cudart
)

set_target_properties(backtest_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Python Module: CUDA Accelerated Backtesting
# ============================================================================
pybind11_add_module(cuda_backtest_module
    bindings/backtest_bindings.cpp
)

target_link_libraries(cuda_backtest_module PRIVATE
    backtest_cuda
    CUDA::cudart
)

# ============================================================================
# Python Module: CUDA Monte Carlo
# ============================================================================
pybind11_add_module(cuda_monte_carlo_module
    bindings/monte_carlo_bindings.cpp
)

target_link_libraries(cuda_monte_carlo_module PRIVATE
    monte_carlo_cuda
    CUDA::cudart
    CUDA::curand
)

# ============================================================================
# Installation
# ============================================================================
install(TARGETS cuda_backtest_module cuda_monte_carlo_module
    LIBRARY DESTINATION python
)

# ============================================================================
# Tests (optional)
# ============================================================================
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_monte_carlo
        tests/test_monte_carlo.cu
    )

    target_link_libraries(test_monte_carlo
        monte_carlo_cuda
    )

    add_test(NAME MonteCarloTest COMMAND test_monte_carlo)
endif()

# ============================================================================
# Build Info
# ============================================================================
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
