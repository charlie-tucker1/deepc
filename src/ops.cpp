
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "deepc/tensor.h"
#include "deepc/ops.h"

#ifdef DEEPC_CUDA
#include "kernels.h"
#endif

namespace deepc {
    class GraphArena;

    namespace { // anon namespace to restrict linkage of cpu compute
        void elwise_add_fwd_cpu(const double *a, const double *b, double *out, int n) {
            for (int i {0}; i < n; i++) {
                out[i] = a[i] + b[i];
            }
        }


        void matmul_fwd_cpu(const double* a, const double* b, double* out, int a_rows, int a_cols, int b_cols) {
            for (int i = 0; i < a_rows; i++) {
                for (int k = 0; k < a_cols; k++) {
                    double av = a[i * a_cols + k];
                    for (int j = 0; j < b_cols; j++)
                        out[i * b_cols + j] += av * b[k * b_cols + j];
                }
            }
        }

        void bias_add_fwd_cpu(const double* a, const double* bias, double* out, const int a_cols, const int a_rows) {

            for (int i = 0; i < a_rows; i++) {
                for (int j = 0; j < a_cols; j++) {
                    out[i * a_cols + j] = a[i * a_cols + j] + bias[j];
                }
            }
        }
    }



    Tensor* add(GraphArena& arena, Tensor* a, Tensor* b) {
        assert(a->rows == b->rows && a->cols == b->cols);
        Tensor* out = arena.make(a->rows, a->cols);
        out->prev = {a, b};
        a->pending++;
        b->pending++;
        int n = a->rows * a->cols;

        elwise_add_fwd_cpu(a->data.get(), b->data.get(), out->data.get(), n);

        out->backward = [a, b, out, n]() {
            for (int i {0}; i < n; i++) {
                a->grad[i] += out->grad[i];
                b->grad[i] += out->grad[i];
            }
        };
        return out;
    }

    Tensor* bias_add(GraphArena& arena, Tensor* a, Tensor* bias) {
        // a: (rows, cols), b: (1, cols) broadcast across every row of a
        assert(bias->rows == 1 && a->cols == bias->cols);
        Tensor* out = arena.make(a->rows, a->cols);
        out->prev = {a, bias};
        a->pending++;
        bias->pending++;

    #ifdef DEEPC_CUDA
            if (arena.backend == Backend::CUDA) {
                bias_add_fwd_gpu(a->data.get(), bias->data.get(), out->data.get(),
                                 a->rows, a->cols);
            } else {
                bias_add_fwd_cpu(a->data.get(), bias->data.get(), out->data.get(),
                                 a->cols, a->rows);
            }
    #else
            bias_add_fwd_cpu(a->data.get(), bias->data.get(), out->data.get(),
                             a->cols, a->rows);
    #endif



        out->backward = [a, bias, out]() {
            for (int i = 0; i < out->rows; i++) {
                for (int j = 0; j < out->cols; j++) {
                    // forward: out[i,j] = a[i,j] + b[0,j]
                    a->grad[i * out->cols + j] += out->grad[i * out->cols + j];
                    bias->grad[j]              += out->grad[i * out->cols + j];  // sum over rows
                }
            }
        };
        return out;
    }

    Tensor * mul(GraphArena& arena, Tensor* a, Tensor* b) {
        assert(a->cols == b->rows);
        Tensor* out = arena.make(a->rows, b->cols);
        out->prev = {a, b};
        a->pending++;
        b->pending++;

#ifdef DEEPC_CUDA
        if (arena.backend == Backend::CUDA) {
            matmul_tiled_fwd_gpu(a->data.get(), b->data.get(), out->data.get(),
                             a->rows, a->cols, b->cols);
        } else {
            matmul_fwd_cpu(a->data.get(), b->data.get(), out->data.get(),
                                        a->rows, a->cols, b->cols);
        }
#else
        matmul_fwd_cpu(a->data.get(), b->data.get(), out->data.get(),
                                        a->rows, a->cols, b->cols);
#endif

        out->backward = [a, b, out]() {

            for (int i = 0; i < a->rows; i++) {
                for (int k = 0; k < a->cols; k++) {
                    for (int j = 0; j < b->cols; j++) {
                        a->grad[i * a->cols + k] += out->grad[i * out->cols + j] * b->data[k * b->cols + j];
                    }
                }
            }
            for (int k = 0; k < b->rows; ++k) {
                for (int j = 0; j < b->cols; j++) {
                    for (int i = 0; i < out->rows; i++) {
                        b->grad[k * b->cols + j] += out->grad[i * out->cols + j] * a->data[i * a->cols + k];
                    }
                }
            }
        };
        return out;
    }


    Tensor * relu(GraphArena& arena, Tensor* a) {
        Tensor* out = arena.make(a->rows, a->cols);
        out->prev = {a};
        a->pending++;
        for (int i = 0; i < a->rows*a->cols; i++) {
            out->data[i] = std::max(a->data[i], 0.0);
        }

        out->backward = [a, out]() {
            for (int i= 0; i < out->rows*out->cols; i++) {
                if (out->data[i] > 0.0) {
                    a->grad[i] += out->grad[i];
                }
            }
        };

        return out;
    }

    Tensor* cross_entropy_loss(GraphArena& arena, Tensor* logits, const std::vector<int>& labels) {
        assert(logits->rows == labels.size());
        Tensor* loss = arena.make(1,1);

        loss->prev = {logits};
        logits->pending++;

        std::vector<double> row_ms;
        std::vector<double> row_psums;

        for (int i = 0; i < labels.size(); ++i) {

            double* start = &logits->data[i * logits->cols];
            double* end = &logits->data[i * logits->cols + logits->cols];
            double m = *std::max_element(start, end);

            row_ms.emplace_back(m);

            double log_softmax = (logits->data[i * logits->cols + labels[i]] - m);
            double psum = 0.0;
            for (int j{0}; j < logits->cols; j++) {
                psum += std::exp(logits->data[i * logits->cols + j] - m);
            }
            loss->data[0] += - (log_softmax - std::log(psum));

            row_psums.emplace_back(psum);
        }

        loss->data[0] /= labels.size();

        loss->backward = [logits, loss, labels, row_psums, row_ms] () {

            for (int i = 0; i < logits->rows; i++)
            {
                for (int j = 0; j < logits->cols; j++) {
                    logits->grad[i*logits->cols + j] += ((std::exp(logits->data[i * logits->cols + j] - row_ms[i]) /
                                            row_psums[i]) - 1*(j==labels[i])) * loss->grad[0] / labels.size();
                }
            }
        };
        return loss;
    }
}