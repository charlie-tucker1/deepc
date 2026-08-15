
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


__global__ void bias_add_kernel_fwd(const double* a, const double* b,
                                double* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i < n)
        out[i] = a[i] + b[i % cols];
}


__global__ void tiled_matrix_multiply_kernel_fwd(const double* a, const double* b, double * out,
                                            int M, int K, int N, int TILE_WIDTH ) {

    //thread specifics
    int t_row = blockDim.y * TILE_WIDTH + threadIdx.y;
    int t_col = blockDim.x * TILE_WIDTH + threadIdx.x;

    //allocate shared mem
    __shared__ double a_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ double b_shared[TILE_WIDTH][TILE_WIDTH];

    //compute proper number of phases
    int n_phases = (K + TILE_WIDTH - 1) / TILE_WIDTH;


    double psum = 0;
    for (int phase = 0; phase < n_phases; phase++) {

        // load shared memory
        a_shared[threadIdx.y][threadIdx.x] = a[t_row * K + (TILE_WIDTH*phase + threadIdx.x)];
        b_shared[threadIdx.y][threadIdx.x] = b[(TILE_WIDTH*phase + threadIdx.y) * N + t_col];
        __syncthreads();

        // compute from shared memory into psum
        for (int k = 0; k < TILE_WIDTH; k++) {
            psum += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }
        __syncthreads();

    }

    out[t_row * N + t_col] = psum;

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

    bias_add_kernel_fwd<<<grid, block>>>(a_d, b_d, out_d, rows, cols);

    CUDA_CHECK(cudaGetLastError());        // catch bad launch config

    CUDA_CHECK(cudaMemcpy(out, out_d, n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(out_d));
}




}











