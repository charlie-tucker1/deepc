#define BIAS_COLS 784
#define BATCH_SIZE 64
#define TILE_WIDTH 32

//Bias add:
// a: (rows, cols) b: (1,cols) broadcast to every row of a

//allocate constant memory to hold bias vector

__constant__ float bias_vec[BIAS_COLS];


//each thread calculates 1 output element
//output dimensions == input dimensions
__global__ void biasAddKernel(float* matrix, float* out) {

    int trow = blockIdx.y * blockDim.y + threadIdx.y;
    int tcol = blockIdx.x * blockDim.x + threadIdx.x;

    if (trow < BATCH_SIZE && tcol < BIAS_COLS) {
        out[trow * BIAS_COLS + tcol] =  matrix[trow * BIAS_COLS + tcol ] + bias_vec[tcol];
    }
}