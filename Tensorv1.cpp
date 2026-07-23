#include <iostream>
#include <vector>
#include <functional>
#include <deque>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <cassert>
#include <random>
#include "mnist_loader.h"
#include <filesystem>
#include <chrono>




struct Tensor {
    inline static int alive {0}; // for debugging / detecting mem leaks

    Tensor(int rows, int cols) : rows{rows}, cols{cols},
        data{std::make_unique<double[]>(rows * cols)},
        grad{std::make_unique<double[]>(rows * cols)}
    {
        alive++;
    }

    int rows, cols;
    std::unique_ptr<double[]> data;
    std::unique_ptr<double[]> grad;

    int pending {0};

    std::vector<Tensor*> prev;
    std::function<void()> backward;

    void init_tensor_random(float minBound, float maxBound) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(minBound, maxBound);
        for (int i = 0; i < this->rows * this->cols; i++) this->data[i] = dist(gen);
    }

    ~Tensor() {
        alive--;
    }
};


class tensorGraphCtx {
public:
    std::vector<std::unique_ptr<Tensor>> tensors; //unique pointers manage graph memory lifespan/ownership

    Tensor* make(int rows, int cols) {
        tensors.emplace_back(std::make_unique<Tensor>(rows, cols));
        return tensors.back().get();
    }

};

Tensor* clone(tensorGraphCtx& ctx, const Tensor* src) {
    Tensor* t = ctx.make(src->rows, src->cols);
    std::copy(src->data.get(), src->data.get() + src->rows*src->cols, t->data.get());
    return t;   // caller reassigns their own variable
}


void elwise_add(const double *a, const double *b, double *out, int n) {
    for (int i {0}; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}


void matmul_host(const double* a, const double* b, double* out, int a_rows, int a_cols, int b_rows, int b_cols) {
    assert(a_cols == b_rows);
    for (int i = 0; i < a_rows; i++) {
        for (int k = 0; k < a_cols; k++) {
            double av = a[i * a_cols + k];
            for (int j = 0; j < b_cols; j++)
                out[i * b_cols + j] += av * b[k * b_cols + j];
        }
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

Tensor* bias_add(tensorGraphCtx& ctx, Tensor* a, Tensor* b) {
    // a: (rows, cols), b: (1, cols) broadcast across every row of a
    assert(b->rows == 1 && a->cols == b->cols);
    Tensor* out = ctx.make(a->rows, a->cols);
    out->prev = {a, b};
    a->pending++;
    b->pending++;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            out->data[i * a->cols + j] = a->data[i * a->cols + j] + b->data[j];
        }
    }

    out->backward = [a, b, out]() {
        for (int i = 0; i < out->rows; i++) {
            for (int j = 0; j < out->cols; j++) {
                // forward: out[i,j] = a[i,j] + b[0,j]
                a->grad[i * out->cols + j] += out->grad[i * out->cols + j];
                b->grad[j]                 += out->grad[i * out->cols + j];  // sum over rows
            }
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

    matmul_host(a->data.get(), b->data.get(), out->data.get(),
                                    a->rows, a->cols, b->rows,b->cols);

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


Tensor * relu(tensorGraphCtx& ctx, Tensor* a) {
    Tensor* out = ctx.make(a->rows, a->cols);
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

Tensor* cross_entropy_loss(tensorGraphCtx& ctx, Tensor* logits, int label) {
    Tensor* loss = ctx.make(1,1);

    loss->prev = {logits};
    logits->pending++;

    double* start = &logits->data[0]; double* end = &logits->data[logits->rows * logits->cols];
    double m = *std::max_element(start, end);

    double log_softmax = (logits->data[label] - m);
    double psum = 0.0;
    for (int j{0}; j < logits->rows * logits->cols; j++) {
        psum += std::exp(logits->data[j] - m);
    }
    loss->data[0] = - (log_softmax - std::log(psum));

    loss->backward = [logits, loss, label, psum, m] () {
        for (int j = 0; j < logits->rows*logits->cols; j++) {
            logits->grad[j] += ((std::exp(logits->data[j] - m) / psum) - 1*(j==label)) * loss->grad[0];
        }
    };
    return loss;
}



void sgd_step(tensorGraphCtx& params, double lr) {
    for (auto& up : params.tensors) {          // up: unique_ptr<Tensor>&
        Tensor* t = up.get();
        int n = t->rows * t->cols;
        for (int i = 0; i < n; i++)
            t->data[i] -= lr * t->grad[i];
    }
}

void zero_grad(tensorGraphCtx& params) {
    for (auto& up : params.tensors) {          // up: unique_ptr<Tensor>&
        Tensor* t = up.get();
        int n = t->rows * t->cols;
        std::fill(t->grad.get(), t->grad.get() + n, 0.0);
    }
}

struct MLP {
    tensorGraphCtx params;        // owns permanent params
    Tensor *W1,  *W2, *b1, *b2;    // raw ptrs into params

    MLP(int in, int hidden, int out) {

        W1 = params.make(in, hidden);  b1 = params.make(1, hidden);
        W2 = params.make(hidden, out); b2 = params.make(1, out);


    }

    // the architecture for our MLP:
    static Tensor* forward_ops(tensorGraphCtx& ctx, Tensor* x,
                               Tensor* W1, Tensor* b1, Tensor* W2, Tensor* b2) {
        Tensor* h = relu(ctx, bias_add(ctx, mul(ctx, x, W1), b1));
        return bias_add(ctx, mul(ctx, h, W2), b2);
    }

    Tensor* forward(tensorGraphCtx& ctx, Tensor* x) {
        return forward_ops(ctx, x, W1, b1, W2, b2);
    }


    int infer(Tensor* x) {
        tensorGraphCtx ctx;
        Tensor* logits = this->forward(ctx, x);
        for (auto& up : params.tensors) up->pending = 0;   // forward-without-backwards reset, as designed
        return std::distance(&logits->data[0],
            std::max_element(&logits->data[0], &logits->data[0]+10));
    }
};




void backwards(Tensor* loss) {
    int n = loss->rows * loss->cols;
    for (int i {0}; i < n; i++) {loss->grad[i] = 1.0;}

    std::deque<Tensor*> ready {loss};
    while (!(ready.empty())) {
        Tensor* t = ready.front(); ready.pop_front();
        if (t->backward) t->backward();
        for (Tensor* v : t->prev) {
            v->pending--;
            if (v->pending == 0) ready.emplace_back(v);
        }
    }
}

struct Graph {
    Tensor* L;
    std::vector<Tensor*> leaves;
};

bool compare_grad_t(double a, double n) {
    const double atol = 1e-8;
    const double rtol = 1e-5;
    return std::abs(a - n) < atol + rtol * std::max(std::abs(a), std::abs(n));
}

bool tensor_gradcheck(std::function<Graph(tensorGraphCtx& ctx, const std::vector<Tensor*>&)> build,
               std::vector<Tensor*> xs, int num_tests)
{

    //No h reaches 16 digits: shrinking h trades truncation for cancellation,
    //and the floor at their crossing is ≈ 3e-11 (~11 digits). h is tuned to the valley bottom, not to eps.

    const double h = 1e-5;                       // step size

    // ---- analytic side:
    tensorGraphCtx an_ctx;
    Graph g = build(an_ctx, xs);              // BUILD CALL. Fresh nodes, pendings
    backwards(g.L);                              // fills every leaf's ->grad via chain rule

    std::random_device rd;
    std::mt19937 gen(rd());

    bool all_ok = true;
    for (size_t i = 0; i < num_tests; ++i) {


        std::uniform_int_distribution<int> dist_leaves(0,xs.size() - 1);
        int nudgeLeaf = dist_leaves(gen);
        std::uniform_int_distribution<int> dist_elements(0,(xs[nudgeLeaf]->rows * xs[nudgeLeaf]->cols) - 1);
        int nudgeIdx = dist_elements(gen);

        double f_plus {0.0};
        {
            tensorGraphCtx plus_ctx;
            std::vector<Tensor*> plus_leaves;
            for (Tensor* x : xs) plus_leaves.emplace_back(clone(plus_ctx, x));

            plus_leaves[nudgeLeaf]->data[nudgeIdx] += h;      // mutate the CLONE, xs untouched

            Graph fpg = build(plus_ctx, plus_leaves);

            for (int i = 0; i < fpg.L->rows * fpg.L->cols; i++) f_plus  += fpg.L->data[i];
        }

        double f_minus {0.0};
        {
            tensorGraphCtx minus_ctx;
            std::vector<Tensor*> minus_leaves;
            for (Tensor* x : xs) minus_leaves.emplace_back(clone(minus_ctx, x));

            minus_leaves[nudgeLeaf]->data[nudgeIdx] -= h;

            Graph fmg = build(minus_ctx, minus_leaves);

            for (int i = 0; i < fmg.L->rows * fmg.L->cols; i++) f_minus  += fmg.L->data[i];
        }


        double numeric  = (f_plus - f_minus) / (2 * h);
        double analytic = g.leaves[nudgeLeaf]->grad[nudgeIdx];

        bool ok = compare_grad_t(analytic, numeric);
        all_ok = all_ok && ok;
        printf("leaf: %d , elem %d:  analytic % .12f   numeric % .12f   %s\n",
               nudgeLeaf, nudgeIdx, analytic, numeric, ok ? "PASS" : "FAIL");
    }
    return all_ok;
}


int main() {


// Load MNIST dataset:

    auto mnist = load_mnist("MNIST_data/train-images.idx3-ubyte",
                            "MNIST_data/train-labels.idx1-ubyte");

    std::cout << "Loaded " << mnist.images.size() << " images\n";
    std::cout << "Rows: " << mnist.rows << ", Cols: " << mnist.cols << "\n";
    std::cout << "First label: " << static_cast<int>(mnist.labels[0]) << "\n";
    std::cout << "First pixel: " << static_cast<int>(mnist.images[0][0]) << "\n";



//Init our MLP with proper dims, randomly init by: +- 1 / sqrt(->dim)
    MLP model(784, 128, 10);
    model.W1->init_tensor_random(-0.04, 0.04);
    model.b1->init_tensor_random(-0.04, 0.04);
    model.W2->init_tensor_random(-0.09, 0.09);
    model.b2->init_tensor_random(-0.09, 0.09);

    int steps = 60000;

std::cout << "Starting training for " << steps << " steps\n";

auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < steps; ++i) {

        tensorGraphCtx step_ctx;
        Tensor* ex_data = step_ctx.make(1,784);

        for (int p = 0; p < 784; ++p)
            ex_data->data[p] = mnist.images[i][p] / 255.0;

        Tensor* logits = model.forward(step_ctx, ex_data);
        Tensor* loss = cross_entropy_loss(step_ctx, logits, static_cast<int>(mnist.labels[i]));


        if (i % 10000 == 0) {std::cout << "loss on step " << i << " was " << loss->data[0] << std::endl;}
        backwards(loss);


        sgd_step(model.params, 0.01);
        zero_grad(model.params);
    }
auto end = std::chrono::steady_clock::now();
auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

std::cout << "Elapsed time: " << elapsed_ms.count() << " ms\n";


    //run inference:


    auto start_infer = std::chrono::steady_clock::now();
    int test_examples = 10000;
    auto mnist_test = load_mnist("MNIST_data/t10k-images.idx3-ubyte",
                            "MNIST_data/t10k-labels.idx1-ubyte");

std::cout << "Starting inference on " << test_examples << " examples\n";

    float accuracy_sum = 0.0;
    for (int i = 0; i < test_examples; ++i) {

        tensorGraphCtx step_ctx;
        Tensor* ex_data = step_ctx.make(1,784);

        for (int p = 0; p < 784; ++p)
            ex_data->data[p] = mnist_test.images[i][p] / 255.0;

        if (model.infer(ex_data) == mnist_test.labels[i]) { ++accuracy_sum; };
    }

    auto end_infer = std::chrono::steady_clock::now();
    auto elapsed_ms_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer);

    std::cout << "Elapsed inference time: " << elapsed_ms_infer.count() << " ms\n";
    std::cout << "Accuracy: " << accuracy_sum / static_cast<float>(test_examples) << "\n";


return 0;
}


