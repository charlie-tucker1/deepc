
//Bias add:
// a: (rows, cols) b: (1,cols) broadcast to every row of a
// each thread calculates 1 output element
// output dimensions == input dimensions

#include <cstdio>
#include "cuda_runtime.h"
#include "../src/kernels.h"


#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(err), __FILE__, __LINE__);        \
        std::abort();                                                \
    }                                                                \
} while (0)


__global__ void bias_add_kernel(const double* a, const double* b,
                                double* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i < n)
        out[i] = a[i] + b[i % cols];
}

namespace deepc {

void bias_add_fwd_gpu(const double* a, const double* b, double* out,
                      int rows, int cols) {
    int n = rows * cols;

    double* out_d, *a_d, *b_d; // device pointers

    CUDA_CHECK(cudaMalloc(&a_d, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b_d, cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&out_d, n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(a_d, a, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, cols * sizeof(double), cudaMemcpyHostToDevice));

    int block = 256;
    int grid  = (n + block - 1) / block;   // ceil-div
    bias_add_kernel<<<grid, block>>>(a_d, b_d, out_d, rows, cols);
    CUDA_CHECK(cudaGetLastError());        // catch bad launch config

    CUDA_CHECK(cudaMemcpy(out, out_d, n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(out_d));
}

}











