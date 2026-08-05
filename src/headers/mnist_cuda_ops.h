//
// Created by charlietucker on 8/5/26.
//

#ifndef DEEPC_MNIST_CUDA_OPS_H
#define DEEPC_MNIST_CUDA_OPS_H
#include <driver_types.h>

#endif //DEEPC_MNIST_CUDA_OPS_H


void launch_bias_add(const float* matrix, const float* bias, float* out,
                     int rows, int cols, cudaStream_t stream = nullptr);