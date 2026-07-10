cmake_minimum_required(VERSION 3.24)
project(deepc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(deepc Value.cpp
        #matmulv0.cu
        matmulv1_general.cu)
target_link_libraries(deepc PRIVATE CUDA::cudart)
