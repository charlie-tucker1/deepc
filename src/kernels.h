#pragma once

#include "cuda_runtime.h"

namespace deepc {
    void bias_add_fwd_gpu(const double* a, const double* b, double* out,
                            int rows, int cols);


}
