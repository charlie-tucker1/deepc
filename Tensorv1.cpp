#include <iostream>
#include <vector>
#include <functional>
#include <deque>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <cassert>





struct Tensor {

    Tensor(int rows, int cols) : rows{rows}, cols{cols},
        data{std::make_unique<double[]>(rows * cols)},
        grad{std::make_unique<double[]>(rows * cols)}
    {
        alive++;
    }


    inline static int alive {0};

    int rows, cols;
    std::unique_ptr<double[]> data;
    std::unique_ptr<double[]> grad;


    int pending {0};

    std::vector<Tensor*> prev;

    std::function<void()> backward;

    ~Tensor() {
        alive--;
    }


};

class tensorGraphCtx {
public:
    std::vector<std::unique_ptr<Tensor>> tensors;

    Tensor* make(int rows, int cols) {
        tensors.emplace_back(std::make_unique<Tensor>(rows, cols));
        return tensors.back().get();
    }
};



void elwise_add(const double *a, const double *b, double *out, int n) {
    for (int i {0}; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

Tensor* add(tensorGraphCtx& ctx, Tensor* a, Tensor* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    Tensor* out = ctx.make(a->rows, a->cols);
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    int n = a->rows * a->cols;
    elwise_add(a->data.get(), b->data.get(), out->data.get(), n);
    out->backward = [a, b, out, n]() {
        for (int i {0}; i < n; i++) {
            a->grad[i] += out->grad[i];
            b->grad[i] += out->grad[i];
        }
    };
    return out;
}

Tensor * mul(tensorGraphCtx& ctx, Tensor* a, Tensor* b) {
    assert(a->cols == b->rows);
    Tensor* out = ctx.make(a->rows, b->cols);
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    for (int i = 0; i < a->rows; i++) {
        for (int k = 0; k < a->cols; k++) {
            double av = a->data[i * a->cols + k];
            for (int j = 0; j < b->cols; j++)
                out->data[i * b->cols + j] += av * b->data[k * b->cols + j];
        }
    }

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


void backwards(Tensor* loss) {
    int n = loss->rows * loss->cols;
    for (int i {0}; i < n; i++) {loss->grad[i] = 1.0;}

    std::deque<Tensor*> ready {loss};
    while (!(ready.empty())) {
        Tensor* t = ready.front(); ready.pop_front();
        if (t->backward) t->backward();
        for (Tensor* v : t->prev) {
            v->pending--;
            if (v->pending == 0) ready.push_back(v);
        }
    }
}

