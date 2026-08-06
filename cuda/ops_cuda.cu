
//Bias add:
// a: (rows, cols) b: (1,cols) broadcast to every row of a
// each thread calculates 1 output element
// output dimensions == input dimensions

namespace {
    constexpr int BX = 32, BY = 8;   // 256 threads; x = 32 → full-warp coalesced columns

    __global__ void biasAddKernel(const float* matrix, const float* bias_vec, float* out,
                                const int BATCH_SIZE, const int BIAS_COLS) {
        int trow = blockIdx.y * blockDim.y + threadIdx.y;
        int tcol = blockIdx.x * blockDim.x + threadIdx.x;

        if (trow < BATCH_SIZE && tcol < BIAS_COLS) {
            out[trow * BIAS_COLS + tcol] =  matrix[trow * BIAS_COLS + tcol ] + bias_vec[tcol];
        }
     }

    constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }
}

void launch_bias_add(const float* matrix, const float* bias, float* out,
                     int rows, int cols, cudaStream_t stream) {
    dim3 block(BX, BY);
    dim3 grid(ceil_div(cols, BX), ceil_div(rows, BY));
    biasAddKernel<<<grid, block, 0, stream>>>(matrix, bias, out, rows, cols);
    //CUDA_CHECK_KERNEL();
}