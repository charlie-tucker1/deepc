#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

__global__ void matMulRowOut(float *A, float *B,float *C,int width) {
    int trow = blockIdx.x * blockDim.x + threadIdx.x;

    if (trow < width) {
        for (int i = 0; i < width; i++) {
            float outVal = 0.0;
            for (int j = 0; j < width; j++) {
                outVal += A[trow* width + j] * B[j * width + i]; // sum(j) over A[trow][j] * B[j][i]
            }
            C[trow * width + i] = outVal;
        }
    }
}

__global__ void matMulColOut(float *A, float *B,float *C,int width) {
    int tcol = blockIdx.y * blockDim.y + threadIdx.y;

    if (tcol < width) {
        for (int i = 0; i < width; i++) {
            float outVal = 0.0;
            for (int j = 0; j < width; j++) {
                outVal += A[i* width + j] * B[j * width + tcol]; // sum(j) over A[i][j] * B[j][tcol]
            }
            C[i * width + tcol] = outVal;
        }
    }
}

__global__ void matmulKernelSquare(float *a, float *b, float *c,
                                    unsigned int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < width) && (col < width)) {
        float cVal {0.0};
        for (int i = 0; i < width; ++i) {
            cVal += a[row*width + i] * b[i*width + col];
        }
        c[row * width + col] = cVal;
    }

}


__global__ void matvec(float * A, float *x, float * out, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < width) {
        float psum =0.0;
        for (int j = 0; j < width; j++) {
            psum += A[tid* width + j] * x[j]; // sum(j) over A[tid][j] * x[j]
        }
        out[tid] = psum;
    }
}

__host__ void init_matrixSquare(float *mat, unsigned int width) {
    for (unsigned int i = 0; i < width*width; i++) {
        mat[i] = rand() % 100;
    }
}

__host__ void init_vec(float *vec, unsigned int width) {
    for (unsigned int i = 0; i < width; i++) {
        vec[i] = rand() % 100;
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
            //std::cout << "i: " << i << " j: " << j << " expected: " << sum << " got: " <<  c[i*N + j] << std::endl;
            assert( sum == c[i*N + j] );
        }
    }
}

int main() {

    //Specify Matrix & Vector sizes (1024)
    int width  = 1024;

    //allocate host memory
    size_t bytes = width * width * sizeof(float);

    float * h_A = (float *)malloc(bytes);
    float * h_B = (float *)malloc(bytes);
    float * h_C = (float *)malloc(bytes);
    float * h_x = (float *)malloc((bytes/width)); // x is our vector here
    float * h_y = (float *)malloc((bytes/width));

    init_matrixSquare(h_A, width);
    init_matrixSquare(h_B, width);
    init_vec(h_x, width);

    // allocate device memory

    float * d_A, *d_B, *d_C, *d_x, *d_y;

    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_x, bytes/width);
    cudaMalloc(&d_y, bytes/width);

    //move matrices and vector to device

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, bytes/width, cudaMemcpyHostToDevice);


    // benchmark + 10 runs

    float TotalTime_Row{0.0f}, TotalTime_Col{0.0f}, TotalTime_El{0.0f};



    //initialize event types for each MM:
    cudaEvent_t start_RowMM, stop_RowMM, start_ColMM, stop_ColMM, start_vecMM, stop_vecMM, start_El_MM, stop_El_MM;

    cudaEventCreate(&start_RowMM);
    cudaEventCreate(&stop_RowMM);
    cudaEventCreate(&start_ColMM);
    cudaEventCreate(&stop_ColMM);
    cudaEventCreate(&start_El_MM);
    cudaEventCreate(&stop_El_MM);
    cudaEventCreate(&start_vecMM);
    cudaEventCreate(&stop_vecMM);



    float ElapsedTime_Ms = 0.0f;


    // Row output MatMul:
    int threads = 32;

    dim3 dimBlock(threads,1,1);
    dim3 dimGrid((width+threads-1) / threads, 1,1);


    for (int n = 0; n < 10; n++) {
        cudaMemset(d_C, 0, bytes);


        cudaEventRecord(start_RowMM, 0); // start recording event Row_MM

        matMulRowOut<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width); //Run Row_MM kernel

        cudaEventRecord(stop_RowMM, 0); //record Stop event

        cudaEventSynchronize(stop_RowMM); // synchronize

        cudaEventElapsedTime(&ElapsedTime_Ms, start_RowMM, stop_RowMM); // get elapsed time Ms

        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost); // send results back to host

        verify_on_CPU_Square(h_A, h_B, h_C, width); //verify results

        std::cout << "Row-MatMul OK\n";
        std::cout << "it took " << ElapsedTime_Ms << " ms for " << n << "th run Row Output MatMul\n";

        TotalTime_Row  += ElapsedTime_Ms;
    }

    std::cout << "Row-MatMul avg over 10 runs" << TotalTime_Row / 10 << " ms\n";

    // Col output MatMul:

    cudaMemset(d_C, 0, bytes); //clear d_C
    memset(h_C, 0, bytes);

    dimBlock = dim3(1, 32, 1);
    dimGrid = dim3(1,(1024 + 32-1) / 32, 1);

    for (int n = 0; n < 10; n++) {
        cudaEventRecord(start_ColMM, 0); // start recording event Col_MM

        matMulColOut<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width); //Run Col_MM kernel

        cudaEventRecord(stop_ColMM, 0); //record Stop event

        cudaEventSynchronize(stop_ColMM); // synchronize

        cudaEventElapsedTime(&ElapsedTime_Ms, start_ColMM, stop_ColMM); // get elapsed time Ms

        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost); // send results back to host

        verify_on_CPU_Square(h_A, h_B, h_C, width); //verify results

        std::cout << "Col-MatMul OK\n";

        std::cout << "it took " << ElapsedTime_Ms << " for the " << n << "th run Col Output MatMul\n";

        TotalTime_Col += ElapsedTime_Ms;
        ElapsedTime_Ms = 0.0f;
    }

    std::cout << "Col-MatMul avg time: " << TotalTime_Col / 10 << " ms\n";

    //Element wise MatMul:
    cudaMemset(d_C, 0, bytes); //clear d_C
    memset(h_C, 0, bytes);

    threads = 32;
    int grid = (width + threads-1) / threads;

    dim3 dimGrid_E(grid, grid, 1);
    dim3 dimBlock_E(threads, threads, 1);

    for (int n = 0; n < 10; n++) {
        cudaEventRecord(start_El_MM, 0); // start recording event El_MM

        matmulKernelSquare<<<dimGrid_E, dimBlock_E>>>(d_A, d_B, d_C, width); //Run El_MM kernel

        cudaEventRecord(stop_El_MM, 0); //record Stop event

        cudaEventSynchronize(stop_El_MM); // synchronize

        cudaEventElapsedTime(&ElapsedTime_Ms, start_El_MM, stop_El_MM); // get elapsed time Ms

        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost); // send results back to host

        verify_on_CPU_Square(h_A, h_B, h_C, width); //verify results

        std::cout << "Elementwise-MatMul OK\n";

        std::cout << "it took " << ElapsedTime_Ms << " for the " << n << "th run of elementwise MatMul\n";

        TotalTime_El += ElapsedTime_Ms;
        ElapsedTime_Ms = 0.0f;
    }

    std::cout << "Elementwise MatMul avg time: " << TotalTime_El / 10 << " ms\n";


    // Vec Mat mul:

    /*
    cudaEventRecord(start_vecMM, 0); // start recording event vec_MM

    matvec<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, width); //Run vec_MM kernel

    cudaEventRecord(stop_vecMM, 0); //record Stop event

    cudaEventSynchronize(stop_vecMM); // synchronize

    cudaEventElapsedTime(&ElapsedTime_Ms, start_vecMM, stop_vecMM); // get elapsed time Ms

    cudaMemcpy(h_y, d_y, bytes/width, cudaMemcpyDeviceToHost); // send results back to host

    verify_on_CPU_Square(h_A, h_x, h_y, width); //verify results

    std::cout << "Row-MatMul OK\n";

    std::cout << "it took " << ElapsedTime_Ms << " for the vector MatMul\n";
    */






    //clean up memory
    cudaEventDestroy(start_RowMM);
    cudaEventDestroy(stop_RowMM);
    cudaEventDestroy(start_ColMM);
    cudaEventDestroy(stop_ColMM);
    cudaEventDestroy(start_El_MM);
    cudaEventDestroy(stop_El_MM);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_x);
    free(h_y);



}

