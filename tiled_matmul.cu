

// Tiled *square* matrix multiplication function

#define TILE_WIDTH 32

__global__ void tiledMatMul(float * a, float * b, float * c, int width) {
    int by = blockIdx.y; int bx = blockIdx.x;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize shared memory arrays:
    __shared__ float M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N[TILE_WIDTH][TILE_WIDTH];

    float pVal {0.0}; // partial sum

    for (int ph {0}; ph < width/TILE_WIDTH; ++ph) {

        M[ty][tx] = a[ (by*TILE_WIDTH + ty)*width + (ph*TILE_WIDTH + tx) ];  // = row*width + ...
        N[ty][tx] = b[ (ph*TILE_WIDTH + ty)*width + (bx*TILE_WIDTH + tx) ];  // = ... + col
        __syncthreads();

        for (int k {0}; k < TILE_WIDTH; ++k) {
            pVal += M[ty * TILE_WIDTH + k] * N[k * TILE_WIDTH + tx];
        }
        __syncthreads();
    }
    c[row * width + col] = pVal;

}
