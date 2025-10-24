cmake_minimum_required(VERSION 3.14)
project(hft_cpp_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for speed
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")

# Find Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find jaxbind (fallback to pybind11 if not available)
find_package(jaxbind CONFIG QUIET)

if(jaxbind_FOUND)
    message(STATUS "Using jaxbind for Python bindings")

    # jaxbind module
    jaxbind_add_module(hft_cpp
        jaxbind/bindings.cpp
    )

    target_include_directories(hft_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)

else()
    message(STATUS "jaxbind not found, falling back to pybind11")
    find_package(pybind11 CONFIG REQUIRED)

    # pybind11 module (compatibility layer)
    pybind11_add_module(cpp_trading
        src/order.cpp
        bindings/all_bindings.cpp
    )

    target_include_directories(cpp_trading PRIVATE ${CMAKE_SOURCE_DIR}/include)

endif()

# Optional: Build standalone tests
option(BUILD_TESTS "Build C++ unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
