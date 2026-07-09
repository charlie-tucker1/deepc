

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

