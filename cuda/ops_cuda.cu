
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


__global__ void bias_add_kernel_fwd(const float* a, const float* b,
                                float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i < n)
        out[i] = a[i] + b[i % cols];
}



__global__ void tiled_matrix_multiply_kernel_fwd(const float* a, const float* b, float * out,
                                            int M, int K, int N) {

    //thread specifics
    int t_row = blockIdx.y * TILE_WIDTH_MM + threadIdx.y;
    int t_col = blockIdx.x * TILE_WIDTH_MM + threadIdx.x;

    //allocate shared mem
    __shared__ float a_shared[TILE_WIDTH_MM][TILE_WIDTH_MM];
    __shared__ float b_shared[TILE_WIDTH_MM][TILE_WIDTH_MM];

    //compute proper number of phases
    int n_phases = (K + TILE_WIDTH_MM - 1) / TILE_WIDTH_MM;


    float psum = 0;
    for (int phase = 0; phase < n_phases; phase++) {

        // load shared memory
        if ((TILE_WIDTH_MM*phase + threadIdx.x) < K && t_row < M) {
            a_shared[threadIdx.y][threadIdx.x] = a[t_row * K + (TILE_WIDTH_MM*phase + threadIdx.x)];
        } else {
            a_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if ((TILE_WIDTH_MM*phase + threadIdx.y) < K && t_col < N) {
            b_shared[threadIdx.y][threadIdx.x] = b[(TILE_WIDTH_MM*phase + threadIdx.y) * N + t_col];
        } else {
            b_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // compute from shared memory into psum

        for (int k = 0; k < TILE_WIDTH_MM; k++) {
            psum += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }
        __syncthreads();

    }

    if (t_row < M && t_col < N) {
        out[t_row * N + t_col] = psum;
    }

}

namespace deepc {

void bias_add_fwd_gpu(const float* a, const float* b, float* out,
                      int rows, int cols) {
    int n = rows * cols;

    float* out_d, *a_d, *b_d; // device pointers

    CUDA_CHECK(cudaMalloc(&a_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_d, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_d, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(a_d, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, cols * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid  = (n + block - 1) / block;   // ceil-div

    bias_add_kernel_fwd<<<grid, block>>>(a_d, b_d, out_d, rows, cols);

    CUDA_CHECK(cudaGetLastError());        // catch bad launch config

    CUDA_CHECK(cudaMemcpy(out, out_d, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(out_d));
}

void matmul_tiled_fwd_gpu(const double* a, const double* b, double* out,
                            int M, int K, int N) {
    int elems_A = M * K;
    int elems_B = N * K;

    double* out_d, *a_d, *b_d; // device pointers

    CUDA_CHECK(cudaMalloc(&a_d, elems_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b_d, elems_B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&out_d, M*N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(a_d, a, elems_A * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, elems_B * sizeof(double), cudaMemcpyHostToDevice));

    dim3 block = dim3(TILE_WIDTH_MM, TILE_WIDTH_MM);
    dim3 grid  = dim3((N + TILE_WIDTH_MM - 1) / TILE_WIDTH_MM,
                        (M + TILE_WIDTH_MM - 1) / TILE_WIDTH_MM);

    tiled_matrix_multiply_kernel_fwd<<<grid, block>>>(a_d, b_d, out_d, M, K, N);

    CUDA_CHECK(cudaGetLastError());        // catch bad launch config

    CUDA_CHECK(cudaMemcpy(out, out_d, M*N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(out_d));

}




}











