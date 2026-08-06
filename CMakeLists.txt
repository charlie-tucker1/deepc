cmake_minimum_required(VERSION 3.24)
project(deepc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- optional CUDA ----
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    find_package(CUDAToolkit REQUIRED)
    set(DEEPC_HAS_CUDA ON)
    message(STATUS "deepc: CUDA enabled (arch: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(DEEPC_HAS_CUDA OFF)
    message(STATUS "deepc: no CUDA compiler found - building CPU-only")
endif()

# ---- engine library ----
add_library(deepc STATIC
        src/tensor.cpp
        src/value.cpp
        src/ops.cpp
        src/mnist_loader.cpp
        src/gradcheck.cpp
        src/utils.cpp
        src/kernels.h
)
target_include_directories(deepc PUBLIC include)
target_compile_definitions(deepc PUBLIC DEEPC_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

if(DEEPC_HAS_CUDA)
    target_sources(deepc PRIVATE cuda/ops_cuda.cu)
    target_link_libraries(deepc PUBLIC CUDA::cudart)
    target_compile_definitions(deepc PUBLIC DEEPC_CUDA=1)
endif()

# ---- executables ----
add_executable(mnist apps/mnist.cpp)
target_link_libraries(mnist PRIVATE deepc)