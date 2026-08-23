#pragma once

#include "cuda_runtime.h"

namespace deepc {
    void bias_add_fwd_gpu(const float* a, const float* b, float* out,
                            int rows, int cols);

    void matmul_tiled_fwd_gpu(const float* a, const float* b, float* out, int M, int K, int N);

}
