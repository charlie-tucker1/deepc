cmake_minimum_required(VERSION 3.24)
project(deepc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(deepc
        src/cpp/mnist_loader.cpp
        src/cpp/Tensorv1.cpp
        src/cpp/Value.cpp
        #src/cuda/pmpp_matmul_v1.cu
        #src/cuda/pmpp_matmul_v0.cu
        src/headers/mnist_loader.h

)
target_link_libraries(deepc PRIVATE CUDA::cudart)
