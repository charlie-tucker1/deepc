// CPU vs GPU correctness harness.
// Each case builds the same graph twice - once on a CPU-backend arena, once on
// a CUDA-backend arena, with identically seeded random inputs - then diffs the
// outputs elementwise. Add a run_case() call per op as GPU kernels come online.

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "deepc/ops.h"
#include "deepc/tensor.h"

namespace {

    int tests_run {0};
    int tests_failed {0};

    void fill_random(deepc::Tensor* t, std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(-10.0, 10.0);
        for (int i = 0; i < t->rows * t->cols; i++)
            t->data[i] = dist(rng);
    }

    // one graph-building recipe, runnable on either backend
    using OpBuilder = std::function<deepc::Tensor*(deepc::GraphArena&, std::mt19937&)>;

    void run_case(const char* name, const OpBuilder& build, double tol = 1e-12) {
        tests_run++;

        std::mt19937 cpu_rng(42), gpu_rng(42); // same seed -> identical inputs

        deepc::GraphArena cpu_arena; // Backend::CPU by default
        deepc::GraphArena gpu_arena;
        gpu_arena.backend = deepc::Backend::CUDA;

        deepc::Tensor* cpu_out = build(cpu_arena, cpu_rng);
        deepc::Tensor* gpu_out = build(gpu_arena, gpu_rng);

        if (cpu_out->rows != gpu_out->rows || cpu_out->cols != gpu_out->cols) {
            tests_failed++;
            printf("FAIL  %-32s shape mismatch: cpu %dx%d vs gpu %dx%d\n", name,
                   cpu_out->rows, cpu_out->cols, gpu_out->rows, gpu_out->cols);
            return;
        }

        int n = cpu_out->rows * cpu_out->cols;
        double max_diff {0.0};
        int first_bad {-1};
        for (int i = 0; i < n; i++) {
            double d = std::fabs(cpu_out->data[i] - gpu_out->data[i]);
            if (d > max_diff) max_diff = d;
            if (d > tol && first_bad < 0) first_bad = i;
        }

        if (first_bad >= 0) {
            tests_failed++;
            printf("FAIL  %-32s max|cpu-gpu| = %.3e, first mismatch [%d,%d]: cpu=%.17g gpu=%.17g\n",
                   name, max_diff,
                   first_bad / cpu_out->cols, first_bad % cpu_out->cols,
                   cpu_out->data[first_bad], gpu_out->data[first_bad]);
        } else {
            printf("pass  %-32s max|cpu-gpu| = %.3e\n", name, max_diff);
        }
    }
}

int main() {
#ifndef DEEPC_CUDA
    printf("deepc was built without CUDA - no GPU backend to compare against.\n");
    return 0;
#else
    struct Shape { int rows, cols; };
    // odd sizes + sizes straddling the 256-thread block boundary
    const std::vector<Shape> shapes {
        {1, 1}, {3, 5}, {1, 300}, {17, 33}, {64, 4},
        {255, 1}, {257, 129}
    };

    for (const Shape& s : shapes) {
        int rows = s.rows, cols = s.cols;
        char name[64];
        snprintf(name, sizeof(name), "bias_add (%dx%d + 1x%d)", rows, cols, cols);
        run_case(name, [rows, cols](deepc::GraphArena& arena, std::mt19937& rng) {
            deepc::Tensor* a = arena.make(rows, cols);
            deepc::Tensor* bias = arena.make(1, cols);
            fill_random(a, rng);
            fill_random(bias, rng);
            return deepc::bias_add(arena, a, bias);
        });
    }

    // template for the next GPU op, e.g. matmul (use a looser tol - summation
    // order differs between backends):
    //
    // run_case("mul (3x5 @ 5x4)", [](deepc::GraphArena& arena, std::mt19937& rng) {
    //     deepc::Tensor* a = arena.make(3, 5);
    //     deepc::Tensor* b = arena.make(5, 4);
    //     fill_random(a, rng);
    //     fill_random(b, rng);
    //     return deepc::mul(arena, a, b);
    // }, 1e-9);

    printf("\n%d/%d cases passed\n", tests_run - tests_failed, tests_run);
    return tests_failed ? 1 : 0;
#endif
}
