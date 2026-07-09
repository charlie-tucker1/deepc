
// N x M
__global__ void matMulRowOut(float *A, float *B,float *C,int width) {
    int trow = blockIdx.x * blockDim.x + threadIdx.x;

    if (trow < width) {
        for (int i = 0; i < width; i++) {
            float outVal = 0.0;
            for (int j = 0; j < width; j++) {
                outVal += A[i * width + j] * B[trow * width + j]; // sum(j) over A[i][j] * B[trow][j]
            }
            C[trow * width + i] = outVal;
        }
    }
}