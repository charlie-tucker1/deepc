#include <cstdlib>
#include <cassert>
#include <iostream>

__global__ void matmulKernelSquare(float *a, float *b, float *c,
                                    unsigned int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < n) && (col < n)) {
        float cVal {0.0};
        for (int i = row; i < n; ++i) {
            cVal += a[row*n + i] * b[n*i + col];
        }
        c[row * n + col] = cVal;
    }

}



__host__ void init_matricesSquare(float *mat, unsigned int bytes) {
    for (unsigned int i = 0; i < bytes*bytes; i++) {
        mat[i] = rand() % 100;
    }
}

__host__ void verify_on_CPU_Square(float *a, float *b, float *c, int N) {
    float sum;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += a[i*N + k] * b[k*N + j];
            }
            assert( sum == c[i*N + j] );
        }
    }
}


int main() {
    int N = 1024;
    size_t bytes = N * N * sizeof(int);


    //Host pointers + memory

    auto *a_h = static_cast<float *>(malloc(bytes));
    auto *b_h = static_cast<float *>(malloc(bytes));
    auto *c_h = static_cast<float *>(malloc(bytes));

    //device pointers:
    float *d_a, *d_b, *d_c;

    //allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //initialize matrices
    init_matrices(a_h, bytes);
    init_matrices(b_h, bytes);

    //copy host mats to device
    cudaMemcpy(d_a, a_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_h, bytes, cudaMemcpyHostToDevice);

    //set thread block and grid dims
    int threads = 16;
    int grid = (N + threads - 1) / threads;

    dim3 dimGrid(grid, grid, 1);
    dim3 dimBlock(threads, threads, 1);


    //call matmul kernel
    matmulKernelSquare<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);


    //send results back to host
    cudaMemcpy(c_h, d_c, bytes, cudaMemcpyDeviceToHost);


    //check result
    verify_on_CPU_Square(a_h, b_h, c_h, N);

    std::cout << "Matrix Multiplication OK" << "\n";
    free(a_h);
    free(b_h);
    free(c_h);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;

}