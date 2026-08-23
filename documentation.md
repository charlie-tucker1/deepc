# deepC — Engine Documentation

> Documents the engine as of commit `14aca2e` (2026-08-22) plus the 2026-08-23 post-review fixes (GPU `float` types + `TILE_WIDTH_MM` restored; `value.cpp` moved out of the library). The complete programs in §13 and the API snippets in §5–8 were compiled (`-Wall`) and run against `libdeepc` built from this tree (CPU backend, MinGW g++ 16). The GPU code was type-checked and linked as host C++ against a CUDA-runtime shim, but has not been compiled with nvcc or run on a GPU — do that on the CUDA machine before trusting it. Where the repository has a known defect, the relevant section says so and links to [Known issues](#12-known-issues-at-this-commit).

---

## Table of contents

1. [Overview](#1-overview)
2. [Repository layout](#2-repository-layout)
3. [Building and running](#3-building-and-running)
4. [Core concepts](#4-core-concepts)
   - 4.1 [Memory model: arenas and two lifetimes](#41-memory-model-arenas-and-two-lifetimes)
   - 4.2 [Graph recording: `prev`, `pending`, `backward`](#42-graph-recording-prev-pending-backward)
   - 4.3 [The backward pass](#43-the-backward-pass)
   - 4.4 [Backends](#44-backends)
5. [API reference: tensors and arenas](#5-api-reference-tensors-and-arenas)
   - 5.1 [`deepc::Tensor`](#51-deepctensor)
   - 5.2 [`deepc::GraphArena`](#52-deepcgrapharena)
   - 5.3 [`deepc::Backend`](#53-deepcbackend)
6. [API reference: ops](#6-api-reference-ops)
   - 6.1 [`add`](#61-add) · 6.2 [`bias_add`](#62-bias_add) · 6.3 [`mul`](#63-mul) · 6.4 [`relu`](#64-relu) · 6.5 [`cross_entropy_loss`](#65-cross_entropy_loss)
7. [API reference: autograd and optimizer utilities](#7-api-reference-autograd-and-optimizer-utilities)
   - 7.1 [`backwards`](#71-backwards) · 7.2 [`sgd_step`](#72-sgd_step) · 7.3 [`zero_grad`](#73-zero_grad) · 7.4 [`clone`](#74-clone)
8. [API reference: gradient checking](#8-api-reference-gradient-checking)
   - 8.1 [`Graph`](#81-graph) · 8.2 [`compare_grad_t`](#82-compare_grad_t) · 8.3 [`tensor_gradcheck`](#83-tensor_gradcheck)
9. [API reference: data loading](#9-api-reference-data-loading)
   - 9.1 [`MNIST`](#91-mnist) · 9.2 [`load_mnist`](#92-load_mnist) · 9.3 [`SupervisedDataset` (stub)](#93-superviseddataset-stub)
10. [API reference: CUDA backend](#10-api-reference-cuda-backend)
11. [Programs](#11-programs)
    - 11.1 [`mnist` — MNIST MLP trainer](#111-mnist--mnist-mlp-trainer)
    - 11.2 [`tests` — CPU vs GPU kernel correctness](#112-tests--cpu-vs-gpu-kernel-correctness)
12. [Known issues at this commit](#12-known-issues-at-this-commit)
13. [Recipes](#13-recipes)
    - 13.1 [Train a model end to end](#131-train-a-model-end-to-end)
    - 13.2 [Add a new op](#132-add-a-new-op)
    - 13.3 [Add a GPU kernel for an existing op](#133-add-a-gpu-kernel-for-an-existing-op)
    - 13.4 [Gradcheck a new op](#134-gradcheck-a-new-op)
14. [Numerical notes](#14-numerical-notes)
15. [Invariants and gotchas (checklist)](#15-invariants-and-gotchas-checklist)
16. [Legacy and scratch files](#16-legacy-and-scratch-files)

---

## 1. Overview

deepC is a from-scratch neural-network training engine in C++17. It has no dependencies beyond the standard library (plus the CUDA runtime when a CUDA compiler is present). It provides:

- A dense, row-major, `float32` 2-D tensor type with a paired gradient buffer.
- Arena-based ownership of tensors (`GraphArena`) so graph teardown is scope exit.
- Eager, define-by-run reverse-mode autodiff: each op computes its forward immediately and records a backward closure.
- Five differentiable ops — `add`, `bias_add`, `mul` (matmul), `relu`, `cross_entropy_loss` — sufficient for an MLP classifier.
- Plain SGD, gradient zeroing, and a central-finite-difference gradient checker.
- An MNIST IDX loader and a reference MNIST MLP trainer (95%+ test accuracy).
- An optional CUDA backend (currently forward-only `bias_add` and tiled matmul) plus a CPU-vs-GPU correctness harness.

Design principles you will see everywhere:

| Principle | Consequence |
|---|---|
| **Arena ownership, raw-pointer references.** | `GraphArena` owns every `Tensor` via `unique_ptr`; everything else holds `Tensor*`. Nothing is ever freed explicitly. |
| **Eager ops with captured backward closures.** | An op is a free function that allocates its output, computes it, and stores a `std::function<void()>` that knows how to push gradient to its inputs. |
| **Pending-count topological traversal.** | `backwards` fires a node's closure only after every consumer of that node has fired, so `+=` accumulation is complete before propagation. |
| **Host-resident data, even on the CUDA backend.** | `Tensor::data` always lives in host memory; GPU kernels copy in, compute, and copy out per call. |
| **`assert` for preconditions.** | Shape checks are `assert`s, so they vanish in Release (`NDEBUG`) builds. Validate shapes yourself in Release. |

---

## 2. Repository layout

```
deepc/
├── CMakeLists.txt              # static lib `deepc` + executables `mnist`, `tests`; CUDA auto-detected
├── README.md                   # project narrative, benchmark table, roadmap
├── documentation.md            # this file
├── include/deepc/              # PUBLIC headers (target_include_directories PUBLIC include)
│   ├── tensor.h                #   Tensor, GraphArena
│   ├── utils.h                 #   Backend enum, clone, sgd_step, zero_grad, backwards
│   ├── ops.h                   #   add, bias_add, mul, relu, cross_entropy_loss
│   ├── gradcheck.h             #   Graph, compare_grad_t, tensor_gradcheck
│   └── mnist_loader.h          #   MNIST, load_mnist   (global namespace)
├── src/
│   ├── tensor.cpp              # Tensor ctor/dtor, init_tensor_random, GraphArena::make
│   ├── ops.cpp                 # CPU kernels (anonymous namespace) + op graph wiring + backward closures
│   ├── utils.cpp               # clone, sgd_step, zero_grad, backwards
│   ├── gradcheck.cpp           # tensor_gradcheck
│   ├── mnist_loader.cpp        # IDX parser
│   ├── kernels.h               # INTERNAL: declarations of GPU host wrappers (bias_add_fwd_gpu, matmul_tiled_fwd_gpu)
│   └── supervised_dataset.cpp  # SupervisedDataset stub (no header yet, not usable)
├── cuda/
│   └── ops_cuda.cu             # CUDA kernels + host wrappers; compiled only when CUDA is found
├── apps/
│   └── mnist.cpp               # MNIST_MLP + training/inference loop  → executable `mnist`
├── tests/
│   └── kernel_correctness.cpp  # CPU-vs-GPU op diff harness            → executable `tests`
├── experiments/                # reference/practice code, NOT part of the build (see §16)
│   ├── pmpp_matmul_v0.cu
│   ├── pmpp_matmul_v1.cu
│   └── value.cpp               # the original scalar autograd prototype
├── tiled_matmul.cu             # scratch square tiled matmul, NOT part of the build (see §16)
└── data/MNIST_data/            # the four MNIST IDX files (tracked in git)
```

Namespaces: everything in the engine is in `namespace deepc` **except** `MNIST` / `load_mnist` (global).

---

## 3. Building and running

deepC is a standard CMake project (minimum CMake 3.24, C++17). It is developed in CLion on Windows and also builds on Linux.

```bash
cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build cmake-build-release
```

### Targets

| Target | Kind | Sources | Notes |
|---|---|---|---|
| `deepc` | static library | `src/*.cpp` (+ `cuda/ops_cuda.cu` when CUDA is enabled) | Public include dir: `include/`. |
| `mnist` | executable | `apps/mnist.cpp` | Trains the MNIST MLP and reports test accuracy. |
| `tests` | executable | `tests/kernel_correctness.cpp` | CPU vs GPU diff harness; exit code 1 on any failure. Currently does not compile on CPU-only machines — see [Known issues](#12-known-issues-at-this-commit). |

There is no `enable_testing()`/`add_test`, so `ctest` finds nothing; run `tests` directly.

### Compile definitions exported by `deepc` (PUBLIC — they propagate to anything that links it)

| Definition | Value | Meaning |
|---|---|---|
| `DEEPC_DATA_DIR` | `"<repo>/data"` | Absolute path to the data directory, baked in at configure time. `apps/mnist.cpp` falls back to the relative `"data"` only if the macro is absent (i.e. when not built through CMake). |
| `DEEPC_CUDA` | `1` | Defined only when a CUDA compiler was found. Guards every GPU code path in `ops.cpp` and the whole body of `tests`. |

### CUDA detection

`CMakeLists.txt` runs `check_language(CUDA)`. If a CUDA compiler is found it enables the language, sets `CMAKE_CUDA_ARCHITECTURES=native` (unless you pass your own), requires `CUDAToolkit`, adds `cuda/ops_cuda.cu` to the library, links `CUDA::cudart`, and defines `DEEPC_CUDA`. Otherwise it prints `deepc: no CUDA compiler found - building CPU-only` and the engine is CPU-only — `GraphArena::backend` can still be set to `Backend::CUDA`, but it is ignored.

To force a CPU-only build on a CUDA machine, configure with `-DCMAKE_CUDA_COMPILER=CMAKE_CUDA_COMPILER-NOTFOUND` (a pre-set value makes `check_language` skip its probe, and a `-NOTFOUND` value is false in `if()`, so the CUDA block is skipped). To pick a specific architecture instead of `native`: e.g. `-DCMAKE_CUDA_ARCHITECTURES=120` for an RTX 50-series card.

### Running

```bash
./cmake-build-release/mnist
```

`mnist` expects the four IDX files under `data/MNIST_data/` (they are tracked in the repo). A Release run prints the loss every 100 batches, per-epoch wall time, then inference time and accuracy:

```
Loaded 60000 images
Rows: 28, Cols: 28
...
loss on batch 0, epoch 0 was 2.31159
...
Elapsed time for epoch 4 : 11506 ms
Starting inference on 10000 examples
Elapsed inference time: 239 ms
Accuracy: 0.9515
```

(The numbers above are from the verification run for this document: g++ 16 `-O3`, batch 64, lr 0.05, 5 epochs. The README's table was measured with a different compiler and `double` tensors; treat the two as separate baselines.)

---

## 4. Core concepts

### 4.1 Memory model: arenas and two lifetimes

A `GraphArena` is a `std::vector<std::unique_ptr<Tensor>>`. `arena.make(rows, cols)` appends a zero-filled tensor and returns a raw pointer that stays valid for as long as the arena lives (the vector holds pointers, so growth never invalidates tensors). There is no `free`: when the arena goes out of scope, every tensor in it is destroyed.

Idiomatic code uses two arenas with different lifetimes:

```cpp
struct Model {
    deepc::GraphArena params;          // persistent: lives as long as the model
    deepc::Tensor *W, *b;
    Model() { W = params.make(784, 10); b = params.make(1, 10); }
};

for (int step = 0; step < n_steps; ++step) {
    deepc::GraphArena step_arena;      // per-step: input, activations, loss
    deepc::Tensor* x = step_arena.make(64, 784);
    // ... fill x, build graph in step_arena using W and b, backwards, sgd_step(params) ...
}                                      // whole graph freed here
```

Rules that follow from this:

- **An op's output goes in the arena you pass to it**, regardless of where its inputs live. Build each step's graph in the step arena so parameters are not duplicated per step.
- **Never let a tensor outlive its arena.** `Tensor*` is a borrowed reference; the arena is the owner.
- **`sgd_step` / `zero_grad` operate on an entire arena.** Keep parameters in an arena that contains *only* parameters.
- `Tensor::alive` counts live tensors across all arenas; assert it returns to the expected value after a scope to detect leaks during development.

### 4.2 Graph recording: `prev`, `pending`, `backward`

Every op does the same four things (see `src/ops.cpp`):

1. `Tensor* out = arena.make(...)` — allocate the output (zero-filled).
2. `out->prev = {inputs...}; input->pending++;` for each input — record the edge and count one more consumer of each input.
3. Compute the forward into `out->data` immediately (eager).
4. `out->backward = [captures]() { input->grad[...] += ...; }` — store a closure that reads `out->grad` and **accumulates** into each input's `grad`.

A tensor created by `make` and never passed through an op is a **leaf**: `prev` empty, `backward` empty, `pending == 0`. Parameters and inputs are leaves.

`pending` is the number of consumers of a tensor in the recorded graph that have not yet propagated gradient into it. It is incremented by ops and decremented by `backwards`.

### 4.3 The backward pass

`backwards(loss)`:

1. Sets **every** element of `loss->grad` to `1.0`. For a `(1,1)` loss that is the usual scalar seed; for a non-scalar tensor it means you get the gradient of **`sum(loss)`** with respect to everything upstream.
2. Runs a worklist starting at `loss`. Popping a node fires its `backward` closure (if any), then decrements `pending` of each node in `prev`; a node is pushed when its `pending` reaches 0.

Because a node fires only after all of its consumers have fired, all `+=` contributions into `node->grad` are complete before `node->backward` reads it. This is why fan-out (one tensor used by several ops) needs no special handling.

Consequences you must respect:

- **Gradients accumulate.** Nothing zeroes `grad` for you. Call `zero_grad(params)` after each `sgd_step` (or before the next forward). Leaves that are re-used across graphs (parameters) otherwise sum gradients across steps.
- **Graphs are single-shot.** After `backwards`, `pending` is 0 along the traversed path. A second `backwards` on the same graph re-fires only the root closure (re-accumulating into its direct inputs), drives the interior counts negative so those closures never fire, and raises no error. Build a fresh graph per step (the per-step arena pattern does this for you).
- **Every consumer of a non-leaf must be an ancestor of `loss`.** If a node `h` is consumed by both the loss path and some side computation that does not feed `loss`, `h->pending` never reaches 0, `h->backward` never fires, and everything upstream of `h` receives **zero** gradient with no error. (Verified: see [Known issues](#12-known-issues-at-this-commit), item 6.) Leaves are unaffected because their gradient is written by their consumers' closures, not by their own.
- **Forward-only passes leave `pending` raised.** Parameters used in an inference forward without a `backwards` keep their incremented counts. This is harmless for leaves (they have no closure), but `MNIST_MLP::infer` resets them to 0 anyway as hygiene; do the same.

### 4.4 Backends

`GraphArena::backend` (default `Backend::CPU`) selects the compute path for ops built in that arena. Today only the forward of `bias_add` and `mul` dispatch on it, and only when the library was compiled with `DEEPC_CUDA`. All backward closures are CPU. Data never lives on the device between ops (see §10).

---

## 5. API reference: tensors and arenas

Header: `include/deepc/tensor.h` (includes `utils.h`).

### 5.1 `deepc::Tensor`

```cpp
struct Tensor {
    inline static int alive {0};          // live Tensor count across the process (debug aid)

    Tensor(int rows, int cols);           // zero-filled data and grad
    ~Tensor();                            // alive--

    int rows, cols;
    std::unique_ptr<float[]> data;        // rows*cols, row-major: data[i*cols + j]
    std::unique_ptr<float[]> grad;        // same shape/layout as data

    int pending {0};                      // outstanding consumers (see §4.2)
    std::vector<Tensor*> prev;            // inputs of the op that produced this tensor
    std::function<void()> backward;       // pushes grad to prev; empty for leaves

    void init_tensor_random(float minBound, float maxBound);
};
```

A dense 2-D `float` matrix plus its gradient. Always create tensors through `GraphArena::make`; the struct is non-copyable and (because it declares a destructor) non-movable, and the engine assumes arena ownership.

| Member | Semantics |
|---|---|
| `rows`, `cols` | Shape. A `(1, n)` tensor is a row vector (used for biases). A `(1,1)` tensor is a scalar (used for losses). |
| `data` | Values, row-major, contiguous, zero-initialised by the constructor. Element `(i, j)` is `data[i * cols + j]`. |
| `grad` | Gradient of the (summed) loss w.r.t. `data`, same layout, zero-initialised. Written by consumers' `backward` closures via `+=`. |
| `pending` | Consumer count; see §4.2. Must be 0 when a graph is built fresh. |
| `prev` | Direct inputs. Empty for leaves. Set by ops. |
| `backward` | Closure set by ops; empty (`!backward`) for leaves. `backwards()` skips nodes whose closure is empty. |
| `alive` | Incremented in the constructor, decremented in the destructor. Not atomic; the engine is single-threaded. |

**`Tensor(int rows, int cols)`** — allocates `rows*cols` floats for `data` and `grad`, both value-initialised to 0. `rows*cols` is computed in `int`. No validation of non-positive sizes.

**`void init_tensor_random(float minBound, float maxBound)`** — fills `data` (not `grad`) with i.i.d. samples from `std::uniform_real_distribution<float>(minBound, maxBound)` (half-open `[min, max)`), using a fresh `std::mt19937` seeded from `std::random_device` on each call. Not seedable, so runs are not reproducible. Used in `mnist.cpp` for the ±1/√fan_in scheme: `W1 ∈ [-0.04, 0.04)` (1/√784 ≈ 0.036), `W2 ∈ [-0.09, 0.09)` (1/√128 ≈ 0.088).

Example:

```cpp
#include "deepc/tensor.h"
using namespace deepc;

GraphArena arena;
Tensor* W = arena.make(784, 128);
W->init_tensor_random(-0.04f, 0.04f);
W->data[0 * W->cols + 5] = 0.0f;          // element (0, 5)
assert(W->pending == 0 && W->prev.empty() && !W->backward);   // a leaf
```

### 5.2 `deepc::GraphArena`

```cpp
class GraphArena {
public:
    std::vector<std::unique_ptr<Tensor>> tensors;   // owns every tensor made here, in creation order
    Backend backend = Backend::CPU;                 // compute path for ops that take this arena
    Tensor* make(int rows, int cols);               // allocate a zero-filled tensor, return borrowed pointer
};
```

Owner of tensors and carrier of the backend choice. Move-only (holds `unique_ptr`s). Destroying the arena destroys all of its tensors; no tensor touches another in its destructor, so destruction order is irrelevant.

| Member | Semantics |
|---|---|
| `tensors` | Public on purpose: `sgd_step` and `zero_grad` iterate it, and `MNIST_MLP::infer` walks it to reset `pending`. Insertion order == creation order. |
| `backend` | Read by `bias_add` and `mul` when compiled with `DEEPC_CUDA`. Set it before building ops; it is not copied onto tensors. |
| `make(rows, cols)` | `emplace_back(std::make_unique<Tensor>(rows, cols))` and return `.get()`. Never returns null (allocation failure throws `std::bad_alloc`). Pointers remain valid for the arena's lifetime. |

Example:

```cpp
GraphArena gpu_arena;
gpu_arena.backend = Backend::CUDA;     // no-op unless built with DEEPC_CUDA
Tensor* a = gpu_arena.make(64, 784);
Tensor* b = gpu_arena.make(784, 128);
Tensor* c = mul(gpu_arena, a, b);      // forward on GPU (if available); c owned by gpu_arena
```

### 5.3 `deepc::Backend`

Header: `include/deepc/utils.h`.

```cpp
enum class Backend { CPU, CUDA };
```

`CUDA` is honoured only when the library was compiled with `DEEPC_CUDA`; otherwise it is silently treated as CPU. There is no runtime query for GPU availability — check `#ifdef DEEPC_CUDA`.

---

## 6. API reference: ops

Header: `include/deepc/ops.h`. Implementation: `src/ops.cpp` (CPU kernels in an anonymous namespace; graph wiring and backward closures in `namespace deepc`).

Common contract for every op:

- Signature shape: `Tensor* op(GraphArena& arena, inputs...)`. The output is allocated in `arena` and returned as a borrowed pointer.
- Inputs may live in any arena (parameters in `params`, activations in the step arena).
- Shape preconditions are `assert`s — compiled out in Release.
- Forward is computed immediately on the CPU, except where noted (`bias_add`, `mul` dispatch on `arena.backend`).
- The backward closure **accumulates** (`+=`) into input `grad`s and runs on the CPU.
- Each op increments `pending` on each input once per use. Passing the same tensor as both inputs (e.g. `add(arena, x, x)`) is allowed; its gradient is correctly doubled.

| Op | Inputs → output shape | Forward | Backend dispatch |
|---|---|---|---|
| `add(a, b)` | `(r,c)`, `(r,c)` → `(r,c)` | `a + b` elementwise | CPU only |
| `bias_add(a, bias)` | `(r,c)`, `(1,c)` → `(r,c)` | `a[i,:] + bias` | CPU / CUDA forward |
| `mul(a, b)` | `(m,k)`, `(k,n)` → `(m,n)` | matrix product | CPU / CUDA forward |
| `relu(a)` | `(r,c)` → `(r,c)` | `max(a, 0)` | CPU only |
| `cross_entropy_loss(logits, labels)` | `(B,C)`, `B` ints → `(1,1)` | mean softmax-NLL over the batch | CPU only |

### 6.1 `add`

```cpp
Tensor* add(GraphArena& arena, Tensor* a, Tensor* b);
```

Elementwise sum of two tensors of identical shape.

- **Precondition:** `assert(a->rows == b->rows && a->cols == b->cols)`.
- **Forward:** `out[i] = a[i] + b[i]` for `i < rows*cols` (`elwise_add_fwd_cpu`).
- **Backward:** `a->grad[i] += out->grad[i]; b->grad[i] += out->grad[i]`.
- **Cost:** O(rows·cols).

```cpp
Tensor* s = add(arena, x, y);   // x, y both (4, 5); s is (4, 5)
```

### 6.2 `bias_add`

```cpp
Tensor* bias_add(GraphArena& arena, Tensor* a, Tensor* bias);
```

Adds a row vector to every row of a matrix (broadcast over rows). This is the bias term of a linear layer.

- **Precondition:** `assert(bias->rows == 1 && a->cols == bias->cols)`.
- **Forward:** `out[i,j] = a[i,j] + bias[0,j]`. On `Backend::CUDA` (with `DEEPC_CUDA`) calls `bias_add_fwd_gpu`, otherwise `bias_add_fwd_cpu`.
- **Backward:** `a->grad[i,j] += out->grad[i,j]`; `bias->grad[0,j] += Σ_i out->grad[i,j]` (sum over the batch dimension).
- **Cost:** O(rows·cols).

```cpp
Tensor* z = bias_add(arena, mul(arena, x, W), b);   // x (B,784), W (784,128), b (1,128) → z (B,128)
```

### 6.3 `mul`

```cpp
Tensor* mul(GraphArena& arena, Tensor* a, Tensor* b);
```

Matrix multiplication `out = a @ b`.

- **Precondition:** `assert(a->cols == b->rows)`. Output is `(a->rows, b->cols)`.
- **Forward (CPU):** `matmul_fwd_cpu` — `i-k-j` loop order, accumulating with `+=` into `out->data`. **Relies on `out` being zero-filled by `make`.** Streams rows of `b`, vectorises well.
- **Forward (CUDA):** `matmul_tiled_fwd_gpu` — 16×16 shared-memory tiled kernel (see §10).
- **Backward:** `a->grad += out->grad @ bᵀ` (loop order i-k-j) and `b->grad += aᵀ @ out->grad` (loop order k-j-i; the inner `i` loop strides by `a->cols` and `out->cols`, which is the "large-stride walk" the README identifies as the CPU training bottleneck).
- **Cost:** O(m·k·n) forward, 2·O(m·k·n) backward.

```cpp
Tensor* x = arena.make(2, 3);  Tensor* W = arena.make(3, 2);
// fill x = [[1,2,3],[4,5,6]], W = [[.5,1],[1.5,2],[2.5,3]]
Tensor* y = mul(arena, x, W);  // y = [[11,14],[24.5,32]]
```

### 6.4 `relu`

```cpp
Tensor* relu(GraphArena& arena, Tensor* a);
```

Elementwise rectifier.

- **Precondition:** none.
- **Forward:** `out[i] = max(a[i], 0.0f)`.
- **Backward:** `a->grad[i] += out->grad[i]` where `out->data[i] > 0`; zero sub-gradient at `a[i] <= 0`. The mask is read from `out->data`, so the closure captures nothing beyond the two pointers.
- **Cost:** O(rows·cols).

### 6.5 `cross_entropy_loss`

```cpp
Tensor* cross_entropy_loss(GraphArena& arena, Tensor* logits, const std::vector<int>& labels);
```

Fused softmax + negative log-likelihood, **averaged over the batch**. Returns a `(1,1)` tensor.

- **Preconditions:** `assert(logits->rows == labels.size())`. Each `labels[i]` must be in `[0, logits->cols)` — **not checked**; an out-of-range label reads out of bounds.
- **Forward:** for each row `i`: `m_i = max_j logits[i,j]`, `psum_i = Σ_j exp(logits[i,j] − m_i)`, `nll_i = −(logits[i, y_i] − m_i − log psum_i)`. `loss = (1/B) Σ_i nll_i`. The max-subtraction makes the exponentials overflow-safe. `m_i` and `psum_i` are stashed **by value** in the closure, as is a copy of `labels`.
- **Backward:** `logits->grad[i,j] += (softmax(logits)[i,j] − 1[j == y_i]) · loss->grad[0] / B`. Softmax is recomputed from `logits->data`, `m_i`, `psum_i`.
- **Why mean, not sum:** the learning rate then means the same thing at every batch size (README §"Ops and autodiff").
- **Cost:** O(B·C) forward and backward.

```cpp
std::vector<int> labels = {3, 0, 7, 7};          // one int per row of logits
Tensor* loss = cross_entropy_loss(step_arena, logits, labels);   // logits (4, 10) → loss (1,1)
std::cout << loss->data[0];                       // mean NLL
backwards(loss);
```

---

## 7. API reference: autograd and optimizer utilities

Header: `include/deepc/utils.h`. Implementation: `src/utils.cpp`.

### 7.1 `backwards`

```cpp
void backwards(Tensor* loss);
```

Reverse-mode pass from `loss` (semantics in §4.3). Seeds all of `loss->grad` with `1.0`, then performs the pending-count traversal. Idempotence: **none** — one call per graph. Complexity: each closure runs once; total cost ≈ the sum of the ops' backward costs.

Preconditions: every node reachable from `loss` has its `pending` equal to the number of consumers it has in the graph (true for a freshly built graph); every consumer of every non-leaf is itself reachable from `loss`.

### 7.2 `sgd_step`

```cpp
void sgd_step(GraphArena& params, float lr);
```

Vanilla SGD over **every tensor in the arena**: `t->data[i] -= lr * t->grad[i]`. No momentum, no weight decay. Does not touch `grad` — follow with `zero_grad`. Pass an arena that contains only parameters; passing a step arena would "train" activations and inputs.

### 7.3 `zero_grad`

```cpp
void zero_grad(GraphArena& params);
```

Sets `grad` to `0.0f` for every tensor in the arena. Required once per step because closures accumulate. Does not reset `pending`.

### 7.4 `clone`

```cpp
Tensor* clone(GraphArena& arena, const Tensor* src);
```

Allocates a new tensor of `src`'s shape in `arena` and copies **`data` only**. The result is a fresh **leaf**: `grad` is zero, `prev` empty, `pending` 0, no `backward`, no edge to `src`. Use it to bring a non-parameter input into a graph without recording a dependency (the gradcheck harness uses it for its ±h copies; `mnist.cpp`'s commented gradcheck uses it to inject the data batch).

```cpp
GraphArena master;  Tensor* x_master = master.make(4, 784);  x_master->init_tensor_random(-1, 1);
GraphArena step;    Tensor* x = clone(step, x_master);   // independent copy, owned by `step`
```

The training and optimizer loop in one place:

```cpp
for (int step = 0; step < n_steps; ++step) {
    GraphArena step_arena;
    Tensor* x = step_arena.make(B, 784);          /* fill x and labels */
    Tensor* loss = cross_entropy_loss(step_arena, model.forward(step_arena, x), labels);
    backwards(loss);                              // fills model.params grads
    sgd_step(model.params, 0.05f);
    zero_grad(model.params);
}
```

---

## 8. API reference: gradient checking

Header: `include/deepc/gradcheck.h`. Implementation: `src/gradcheck.cpp`.

> **Status at this commit:** the harness is calibrated for `double` and **does not pass with the current `float` engine** — see [Known issues](#12-known-issues-at-this-commit), item 3. The contract below is still the contract to write against.

### 8.1 `Graph`

```cpp
struct Graph {
    Tensor* L;                     // the graph's output (any shape; the harness sums its elements)
    std::vector<Tensor*> leaves;   // the leaves whose gradients are to be checked, in caller order
};
```

Returned by a build lambda to tell the harness which tensor is the loss and which leaves' `grad`s to read.

### 8.2 `compare_grad_t`

```cpp
bool compare_grad_t(float a, float n);
```

`|a − n| < atol + rtol · max(|a|, |n|)` with `atol = 1e-8`, `rtol = 1e-5`. Exposed so tests can reuse the verdict rule.

### 8.3 `tensor_gradcheck`

```cpp
bool tensor_gradcheck(std::function<Graph(GraphArena& arena, const std::vector<Tensor*>&)> build,
                      std::vector<Tensor*> leaves, int num_tests);
```

Checks analytic gradients against central finite differences for `num_tests` randomly chosen `(leaf, element)` pairs. Prints one line per test (`leaf: k , elem i:  analytic …  numeric …  PASS|FAIL`) and returns `true` only if all pass.

Algorithm:

1. **Analytic pass.** Creates an arena, calls `build(arena, leaves)` with the **caller's real leaves**, runs `backwards(g.L)`. The leaves' `grad` buffers now hold the analytic gradient of `sum(L)`.
2. **Numeric passes.** For each test: pick a random leaf and element; in a fresh arena, `clone` every leaf, add `h = 1e-5` to the chosen element of the chosen clone, `build` on the clones, and read `f(+h) = Σ L`; repeat with `−h`. `numeric = (f(+h) − f(−h)) / 2h`.
3. Compare with `compare_grad_t(analytic, numeric)`.

Contract for the `build` lambda (violating any of these produces FAILs or meaningless output):

| Rule | Why |
|---|---|
| **Use the `leaves` argument directly** — do not clone them inside `build`. | The analytic pass must wire the caller's tensors into the graph so `backwards` writes into the `grad` buffers the harness reads. Symptom of cloning: analytic gradients exactly 0. |
| **Be deterministic in `leaves`.** No fresh randomness inside `build`. | f(+h) and f(−h) must be the same function. Hoist random data out, and `clone` it *in* (the data tensor is then not a leaf). |
| **Leaves must be distinct tensors.** | The numeric side clones each entry separately and nudges one; duplicates would be treated as independent there but as one tensor analytically. |
| **Leaves must have zero `grad` before the call.** | The harness never zeroes them; `backwards` accumulates. Running two gradchecks on the same leaves, or gradchecking after a training step, pollutes the analytic column. |
| **Return `Graph{L, leaves}` with the same leaf order you received.** | `g.leaves[k]` is indexed by the same `k` used to pick the nudged clone. |

Example (the shape used for the MLP, as in the commented block of `apps/mnist.cpp`):

```cpp
#include "deepc/gradcheck.h"
#include "deepc/ops.h"
#include "deepc/utils.h"

GraphArena params;
Tensor* W1 = params.make(784, 128); W1->init_tensor_random(-0.04f, 0.04f);
Tensor* b1 = params.make(1, 128);   b1->init_tensor_random(-0.04f, 0.04f);
Tensor* W2 = params.make(128, 10);  W2->init_tensor_random(-0.09f, 0.09f);
Tensor* b2 = params.make(1, 10);    b2->init_tensor_random(-0.09f, 0.09f);

GraphArena x_arena;                               // outlives the gradcheck
Tensor* x_master = x_arena.make(4, 784);
x_master->init_tensor_random(-1.0f, 1.0f);
std::vector<int> labels = {0, 4, 2, 1};

auto build = [labels, x_master](GraphArena& arena, const std::vector<Tensor*>& leaves) {
    Tensor* x = clone(arena, x_master);           // non-leaf input: cloned IN, deterministic
    Tensor* h = relu(arena, bias_add(arena, mul(arena, x, leaves[0]), leaves[1]));
    Tensor* logits = bias_add(arena, mul(arena, h, leaves[2]), leaves[3]);
    return Graph{cross_entropy_loss(arena, logits, labels), leaves};
};

bool ok = tensor_gradcheck(build, {W1, b1, W2, b2}, 30);
```

---

## 9. API reference: data loading

Header: `include/deepc/mnist_loader.h`. Implementation: `src/mnist_loader.cpp`. **Global namespace** (not `deepc`).

### 9.1 `MNIST`

```cpp
struct MNIST {
    std::vector<std::vector<uint8_t>> images;   // images[i] has rows*cols bytes, row-major, 0..255
    std::vector<uint8_t> labels;                // labels[i] in 0..9
    int rows = 0, cols = 0;                     // 28, 28 for MNIST
};
```

Raw, unnormalised image bytes. Conversion to `float` and scaling (`/ 255.0`) is the caller's job (see `apps/mnist.cpp`).

### 9.2 `load_mnist`

```cpp
MNIST load_mnist(const std::string& image_path, const std::string& label_path);
```

Parses a pair of IDX files (big-endian headers): magic `2051` for images (`idx3-ubyte`), `2049` for labels (`idx1-ubyte`).

- Throws `std::runtime_error` with one of: `"Could not open MNIST files"`, `"Bad image magic"`, `"Bad label magic"`, `"Image/label count mismatch"`.
- Does not detect truncated files: a short `read` is not checked, and the unread tail of an image stays at the zeros `resize` produced.
- Cost: one buffered `read` per image plus one per label; the whole 60k training set is loaded before training starts.

```cpp
const MNIST train = load_mnist(DEEPC_DATA_DIR "/MNIST_data/train-images.idx3-ubyte",
                               DEEPC_DATA_DIR "/MNIST_data/train-labels.idx1-ubyte");
Tensor* x = step_arena.make(B, 784);
for (int j = 0; j < B; ++j)
    for (int p = 0; p < 784; ++p)
        x->data[j * 784 + p] = train.images[first + j][p] / 255.0f;
```

### 9.3 `SupervisedDataset` (stub)

`src/supervised_dataset.cpp` currently contains only a private-member sketch and has no header, so nothing can use it yet:

```cpp
namespace deepc {
    class SupervisedDataset {
        GraphArena example_data;
        std::vector<int> labels;
    };
}
```

It is compiled into `libdeepc` but exposes no API. Points to keep in mind while designing it (derived from the engine's current rules, §4):

- Tensors have no view/slice concept (`data` is an owning `unique_ptr<float[]>`), so a batch must be **copied** into a step-arena tensor, exactly as `mnist.cpp` does today, unless a non-owning view type is introduced.
- If examples live in a `GraphArena`, never pass that arena to `sgd_step`/`zero_grad`, and remember ops will bump `pending` on any dataset tensor they consume directly.
- `cross_entropy_loss` takes labels as `std::vector<int>` per batch; a dataset that stores labels as `int` avoids a per-batch conversion.

---

## 10. API reference: CUDA backend

Files: `src/kernels.h` (internal declarations; **not** under `include/`), `cuda/ops_cuda.cu` (kernels + host wrappers). Compiled only when CMake finds a CUDA compiler. Everything here is an implementation detail of `bias_add` and `mul`; user code never calls these directly.

> **Status:** the compile breaks introduced by the doubles→floats commit (undefined `TILE_WIDTH_MM`; `double`/`float` mismatch) were fixed on 2026-08-23 ([Known issues](#12-known-issues-at-this-commit), item 1). The chain — `ops.cpp` call site, `kernels.h` declaration, wrapper definition, kernel arguments — was verified to agree by compiling and linking it as host C++ against a CUDA-runtime shim; it has **not** yet been compiled with nvcc or run on a GPU.

### Data-movement model

Tensors are host-resident. Each GPU host wrapper, per call: `cudaMalloc` device buffers → `cudaMemcpy` inputs H→D → launch kernel → `cudaMemcpy` output D→H → `cudaFree`. This makes the GPU path correct but slow (three allocations and two transfers per op), which is fine for the current goal (kernel correctness) and is the obvious next thing to change (device-resident tensors, persistent buffers).

Every CUDA runtime call is wrapped in `CUDA_CHECK(call)`, which prints `CUDA error <msg> at <file>:<line>` to `stderr` and `std::abort()`s on failure. `cudaGetLastError()` is checked right after each launch to catch launch-configuration errors; because the following `cudaMemcpy` is synchronous, asynchronous kernel faults surface there.

### `bias_add_fwd_gpu`

```cpp
void bias_add_fwd_gpu(const float* a, const float* b, float* out, int rows, int cols);
```

Host wrapper for `bias_add_kernel_fwd`: 1-D grid, 256 threads per block, `ceil(rows*cols / 256)` blocks; thread `i` computes `out[i] = a[i] + b[i % cols]`. Bit-identical to the CPU path (one addition per element).

### `matmul_tiled_fwd_gpu`

```cpp
void matmul_tiled_fwd_gpu(const float* a, const float* b, float* out, int M, int K, int N);
```

Host wrapper for `tiled_matrix_multiply_kernel_fwd`, computing `out(M×N) = a(M×K) · b(K×N)`:

- Block = `TILE_WIDTH_MM × TILE_WIDTH_MM` = 16×16 threads; grid = `ceil(N/T) × ceil(M/T)`.
- Each block walks `ceil(K/T)` phases; in each phase it loads one `T×T` tile of `a` and of `b` into `__shared__` memory (zero-padding out-of-range elements, so any M, K, N work), syncs, accumulates `T` products per thread into a register `psum`, syncs.
- Threads outside the output bounds still participate in loads/syncs (required for `__syncthreads` correctness) but skip the final store.
- Per output element the `k` accumulation order is sequential and identical to the CPU `i-k-j` loop; the only CPU/GPU numeric difference is nvcc's default FMA contraction (`--fmad=true`), measured at ≈2.4e-4 absolute / ≈1e-4 relative on outputs of magnitude ~1e3 at the test shapes (§14).

`ops.cpp` calls it as `(a, b, out, a->rows, a->cols, b->cols)` = `(M, K, N)`; the declaration in `kernels.h` and the `.cu` definition both use that parameter order.

---

## 11. Programs

### 11.1 `mnist` — MNIST MLP trainer

Source: `apps/mnist.cpp`. Links `deepc`. Defines `deepc::MNIST_MLP` and `main`.

**`MNIST_MLP`**

```cpp
struct MNIST_MLP {
    GraphArena params;                  // owns W1, b1, W2, b2
    Tensor *W1, *W2, *b1, *b2;

    MNIST_MLP(int in, int hidden, int out);     // W1 (in,hidden), b1 (1,hidden), W2 (hidden,out), b2 (1,out); all zero

    static Tensor* mnist_forward_ops(GraphArena& arena, Tensor* x,
                                     Tensor* W1, Tensor* b1, Tensor* W2, Tensor* b2);
    // relu(bias_add(mul(x, W1), b1)) → bias_add(mul(h, W2), b2); returns logits (x->rows, out)

    Tensor* forward(GraphArena& arena, Tensor* x);   // mnist_forward_ops with this model's params
    int infer(Tensor* x);                            // forward in a local arena, reset params' pending, argmax of row 0
};
```

`mnist_forward_ops` is `static` and takes the parameters explicitly so the same architecture can be gradchecked with harness-supplied leaves (see the commented block in the file and §8.3). `infer` hard-codes the argmax over 10 columns (it should use `logits->cols`; [Known issues](#12-known-issues-at-this-commit), item 8).

**`main` — what it does, in order**

1. `load_mnist` on the training pair; print count, shape, first label/pixel.
2. Construct `MNIST_MLP(784, 128, 10)`; init `W1, b1 ∈ [-0.04, 0.04)`, `W2, b2 ∈ [-0.09, 0.09)`.
3. Train: `epochs = 5`, `batch_size = 64`, `steps = 60000` ⇒ `steps / batch_size = 937` batches per epoch (the last 32 examples are never used), fixed order, no shuffle. Per batch: new `step_arena`, copy pixels `/255.0` into a `(64, 784)` tensor, `forward`, `cross_entropy_loss`, `backwards`, `sgd_step(params, 0.05)`, `zero_grad(params)`. Loss printed every 100 batches; epoch time printed.
4. Load the test pair and run `infer` on each of the 10 000 test images one at a time; print inference time and accuracy.

Hyperparameters are literals in `main` (`steps`, `batch_size`, `epochs`, learning rate `0.05`, init bounds). There is no CLI.

### 11.2 `tests` — CPU vs GPU kernel correctness

Source: `tests/kernel_correctness.cpp`. Links `deepc`. Without `DEEPC_CUDA` the whole body is compiled out and `main` prints `deepc was built without CUDA - no GPU backend to compare against.` and returns 0 (but see [Known issues](#12-known-issues-at-this-commit), item 2: the file currently fails to compile in that configuration).

**Harness**

```cpp
using OpBuilder = std::function<deepc::Tensor*(deepc::GraphArena&, std::mt19937&)>;
void run_case(const char* name, const OpBuilder& build, float tol = 1e-12);
```

`run_case` builds the same graph twice — once in a `Backend::CPU` arena, once in a `Backend::CUDA` arena — feeding each build an `mt19937` seeded with 42 so inputs are identical, then compares the two outputs elementwise. Reports `pass` with `max|cpu-gpu|`, or `FAIL` with the first offending index and both values (`%.17g`), or `FAIL … shape mismatch`. Counts are printed at the end; exit code is 1 if anything failed.

`fill_random(t, rng)` fills a tensor from `uniform_real_distribution<float>(-10, 10)` using the supplied generator (deterministic, unlike `Tensor::init_tensor_random`).

**Cases at this commit**

- `bias_add` at shapes `{1,1} {3,5} {1,300} {17,33} {64,4} {255,1} {257,129}` — odd sizes and sizes straddling the 256-thread block boundary; default `tol = 1e-12` (exact agreement is expected for a single addition).
- `mul` ("tiled matmul") at `42×80×74`, `64×64×64`, `64×128×10`, `33×17×10` with `tol = 1e-9` (too tight for `float`; item 4 in Known issues).

**Adding a case**

```cpp
run_case("relu (17x33)", [](deepc::GraphArena& arena, std::mt19937& rng) {
    deepc::Tensor* a = arena.make(17, 33);
    fill_random(a, rng);
    return deepc::relu(arena, a);
}, /*tol=*/0.0f);
```

---

## 12. Known issues at this commit

Found while writing this document; each was verified by compilation, by running code, or by reading the diff of commit `354d418` ("swap doubles for floats"). Numbered so the sections above can point at them. Items 1 and 9 (and the `<cstdlib>` part of 10) were **fixed on 2026-08-23**; the entries are kept, struck through, so the numbering the rest of this document references stays stable.

1. ~~**CUDA build is broken (three compile errors in the GPU path).** Commit `354d418` deleted `#define TILE_WIDTH_MM 16` from `cuda/ops_cuda.cu` but the kernel and host wrapper still use it; `matmul_tiled_fwd_gpu` (both the `.cu` definition and the `src/kernels.h` declaration) was left as `double*` while the kernel it launches and the `Tensor` data it receives are `float*`.~~ **Fixed 2026-08-23**: `#define TILE_WIDTH_MM 16` restored, wrapper + declaration switched to `float*`, `<cstdlib>` added for `std::abort`. Verified by host-C++ shim compile/link only — still needs an nvcc build and a `tests` run on the GPU machine.
2. **`tests` does not compile on CPU-only machines.** `tests/kernel_correctness.cpp` includes `../src/kernels.h`, which unconditionally includes `cuda_runtime.h`. The include is unnecessary (the test only uses `ops.h`/`tensor.h`), and `kernels.h` itself does not need `cuda_runtime.h`.
3. **`tensor_gradcheck` fails for every op on the `float` engine.** `h = 1e-5` and `rtol = 1e-5` are a double-precision calibration. In fp32 the finite-difference error bottoms out near `h ≈ 1e-2` (measured ≈7e-6 relative) and is ≈9e-2 at `h = 1e-5`; even `add` (true gradient exactly 1) reads 1.00136. The README's "every op is certified" statement cannot currently be reproduced. Options: build the gradcheck path with `double` tensors (e.g. a `Scalar` typedef selected at compile time), or recalibrate for fp32 (`h ≈ 1e-2·max(1,|x|)`, `rtol ≈ 1e-3`, `atol ≈ 1e-4`) and accept a weaker certificate.
4. **`tests` matmul tolerance is unreachable in fp32.** With identical per-element summation order the only CPU/GPU difference is FMA contraction, measured at ≈2.4e-4 absolute at the test shapes — five orders of magnitude above `1e-9`. Either compare relative error at ~1e-4–1e-3, or compile the kernel with `--fmad=false` (then agreement is bit-exact).
5. **`tensor_gradcheck` does not zero leaf gradients before the analytic pass**, so the analytic column accumulates across calls on the same leaves.
6. **`backwards` silently drops gradient when a non-leaf has a consumer outside the loss's ancestry** (§4.3). Verified: `h = relu(w); side = add(h, v); L = ce(h)` leaves `h->pending == 1`, `h->backward` never fires, `w->grad == 0`. Not triggered by the MNIST app; will matter for multi-output graphs. A fix is to count only consumers reachable from `loss` (one reachability pass before the traversal) or to use a DFS topological order.
7. **`README.md` run instruction is stale** — it says `./cmake-build-release/deepc`; the executable is now `mnist`.
8. **`MNIST_MLP::infer` hard-codes 10 classes** in the argmax instead of `logits->cols`.
9. ~~**`src/value.cpp` (legacy scalar prototype) is compiled into `libdeepc`** with global-namespace external symbols (`mul`, `add`, `exp`, `log`, `pow`, `tanh`, `relu`, `div`, `sub`, `backwards`, `gradcheck`, `compare_grad`, …). Harmless today, but a collision risk for any consumer.~~ **Fixed 2026-08-23**: moved to `experiments/value.cpp` and removed from the library sources; its symbols are no longer in `libdeepc` (checked with `nm`).
10. Minor: the commented-out gradcheck block in `apps/mnist.cpp` uses an old type name (`tensorGrapharena`); `src/utils.cpp` and `src/gradcheck.cpp` rely on transitive includes for `<algorithm>`/`<cstdio>`. (~~`cuda/ops_cuda.cu` uses `std::abort` without `<cstdlib>`~~ — fixed with item 1.)

---

## 13. Recipes

### 13.1 Train a model end to end

A complete, self-contained program (synthetic data, so it runs without MNIST):

```cpp
#include <vector>
#include <cstdio>
#include "deepc/tensor.h"
#include "deepc/ops.h"
#include "deepc/utils.h"
using namespace deepc;

struct TwoLayer {
    GraphArena params;
    Tensor *W1, *b1, *W2, *b2;
    TwoLayer(int in, int hidden, int out) {
        W1 = params.make(in, hidden);     W1->init_tensor_random(-0.1f, 0.1f);
        b1 = params.make(1, hidden);
        W2 = params.make(hidden, out);    W2->init_tensor_random(-0.1f, 0.1f);
        b2 = params.make(1, out);
    }
    Tensor* forward(GraphArena& a, Tensor* x) {
        Tensor* h = relu(a, bias_add(a, mul(a, x, W1), b1));
        return bias_add(a, mul(a, h, W2), b2);
    }
};

int main() {
    const int B = 32, D = 20, C = 4;
    TwoLayer model(D, 64, C);

    for (int step = 0; step < 500; ++step) {
        GraphArena step_arena;                       // per-step graph lifetime
        Tensor* x = step_arena.make(B, D);
        x->init_tensor_random(-1.0f, 1.0f);
        std::vector<int> y(B);
        for (int i = 0; i < B; ++i) {                // toy task: class = argmax of first C features
            int best = 0;
            for (int j = 1; j < C; ++j) if (x->data[i*D + j] > x->data[i*D + best]) best = j;
            y[i] = best;
        }
        Tensor* loss = cross_entropy_loss(step_arena, model.forward(step_arena, x), y);
        backwards(loss);
        sgd_step(model.params, 0.1f);
        zero_grad(model.params);
        if (step % 100 == 0) printf("step %d loss %.4f\n", step, loss->data[0]);
    }                                                // step_arena and its graph freed here
    return 0;
}
```

### 13.2 Add a new op

Follow the pattern in `src/ops.cpp` exactly; the four steps are the contract `backwards` depends on.

```cpp
// ops.h
Tensor* scale(GraphArena& arena, Tensor* a, float s);

// ops.cpp
Tensor* scale(GraphArena& arena, Tensor* a, float s) {
    Tensor* out = arena.make(a->rows, a->cols);          // 1. allocate output in caller's arena
    out->prev = {a};                                     // 2. record edge …
    a->pending++;                                        //    … and one more consumer of a
    const int n = a->rows * a->cols;
    for (int i = 0; i < n; ++i) out->data[i] = s * a->data[i];   // 3. eager forward
    out->backward = [a, out, s, n]() {                   // 4. backward: ACCUMULATE into inputs
        for (int i = 0; i < n; ++i) a->grad[i] += s * out->grad[i];
    };
    return out;
}
```

Checklist: `assert` shape preconditions; capture by value anything the backward needs that could change (the CE op stashes per-row `m`/`psum`); never write `=` into an input's `grad`; add a gradcheck (§13.4); if the op is a reduction, remember `backwards` seeds every element of `loss->grad`.

### 13.3 Add a GPU kernel for an existing op

1. Declare a host wrapper in `src/kernels.h`: `void relu_fwd_gpu(const float* a, float* out, int n);`
2. Implement the kernel and wrapper in `cuda/ops_cuda.cu` inside `namespace deepc`, using `CUDA_CHECK` around every runtime call and `cudaGetLastError()` after the launch. Follow the existing wrappers' malloc → memcpy → launch → memcpy → free shape.
3. Dispatch in `src/ops.cpp` under `#ifdef DEEPC_CUDA` on `arena.backend == Backend::CUDA`, keeping the CPU call in the `#else`/fallback branch.
4. Add a `run_case` in `tests/kernel_correctness.cpp` (§11.2) with a tolerance appropriate to fp32 (exact for elementwise ops; relative ~1e-4–1e-3 for reductions, or `--fmad=false`).

### 13.4 Gradcheck a new op

```cpp
GraphArena leaf_arena;
Tensor* a = leaf_arena.make(4, 5); a->init_tensor_random(-1.0f, 1.0f);   // grad is zero: fresh tensor

auto build = [](GraphArena& arena, const std::vector<Tensor*>& leaves) {
    return Graph{ scale(arena, leaves[0], 3.0f), leaves };   // use leaves directly, no randomness
};
bool ok = tensor_gradcheck(build, {a}, /*num_tests=*/10);
```

Keep every intermediate O(1) in magnitude (inputs in `[-1, 1]`) so the finite-difference ruler works at its best; the formula being certified is scale-free, so certification at tame scale transfers. See item 3 in Known issues for the fp32 caveat.

---

## 14. Numerical notes

- **Precision.** All storage and arithmetic is `float` (fp32, ε ≈ 1.19e-7). Accumulations (`matmul_fwd_cpu`, the CE row sums, the batch mean) are fp32 too. The README benchmark table predates the switch.
- **Finite differences in fp32.** Measured on the MLP graph with inputs in `[-1, 1]`: relative error of the central difference vs. the analytic gradient was 4e-5 (`h=1e-1`), 7e-6 (`1e-2`), 3e-4 (`1e-3`), 7e-4 (`1e-4`), 9e-2 (`1e-5`), 1e-2 (`1e-6`). The sweet spot is `h ≈ 1e-2`, and the attainable agreement is ~1e-5–1e-4, not 1e-11 as in double.
- **CPU vs GPU matmul.** Same per-element `k` order on both; differences come only from FMA contraction: ≈2.4e-4 absolute (≈1e-4 relative) at the `tests` shapes with inputs in `[-10, 10]` and outputs ~1e3.
- **Cross-entropy stability.** Row max subtraction before `exp` prevents overflow; `log(psum)` with `psum ≥ 1` is always finite.
- **Initialisation.** Uniform ±1/√fan_in (`0.04` for 784, `0.09` for 128) keeps pre-activations O(1) at init; biases are initialised with the same bounds in `mnist.cpp`.
- **Batch mean in the loss** makes `lr` comparable across batch sizes; the README documents the measured lr×batch coupling (lr 0.05 unlocked 95% at B=64).

---

## 15. Invariants and gotchas (checklist)

- `make` zero-fills `data` and `grad`; `mul`'s CPU forward **relies** on `out` being zero.
- `Tensor*` is borrowed; the `GraphArena` owns it. Don't use a pointer after its arena is destroyed.
- Put each step's graph in a fresh arena; never call `backwards` twice on one graph.
- `backwards` computes the gradient of `sum(loss)`; for a `(1,1)` loss that is the loss itself.
- Gradients accumulate: `zero_grad(params)` every step.
- `sgd_step` / `zero_grad` touch **every** tensor in the arena they're given — keep parameters in a parameters-only arena.
- Every consumer of a non-leaf must be reachable from `loss`, or upstream gradients are silently zero.
- After a forward without `backwards` (inference), reset `pending` on the shared leaves (as `MNIST_MLP::infer` does).
- Shape preconditions are `assert`s: absent in Release builds. Labels passed to `cross_entropy_loss` are never range-checked.
- `init_tensor_random` is not seedable; use your own RNG (as `tests` does) when you need reproducibility.
- `Backend::CUDA` is ignored unless the library was compiled with `DEEPC_CUDA`; even then, only `bias_add` and `mul` forwards use it, and data round-trips host↔device per op.
- `gradcheck` build lambdas: use the passed leaves, be deterministic, leaves distinct, leaves' `grad` zero.
- `MNIST`/`load_mnist` are in the global namespace; everything else is in `deepc`.
- Single-threaded throughout (`Tensor::alive` is a plain static).

---

## 16. Legacy and scratch files

These are tracked in the repository but are not part of the engine's API.

| File | What it is | Build status |
|---|---|---|
| `experiments/value.cpp` | The original scalar autograd prototype (`struct value`, `GraphContext`, scalar `mul/add/sub/exp/div/log/pow/tanh/relu`, scalar `backwards`, scalar `gradcheck`). The tensor engine is a direct generalisation of it; the pending-count `backwards` and the gradcheck design were developed here. Its `main` is commented out (and references a `Value` type that no longer exists). | Not in the build (moved out of the library 2026-08-23 so its global-namespace symbols can't collide). |
| `tiled_matmul.cu` | Scratch square-only shared-memory tiled matmul (`TILE_WIDTH 32`), precursor of the kernel in `cuda/ops_cuda.cu`. Indexes the 2-D shared arrays with 1-D syntax and would not compile as-is. | Not in the build. |
| `experiments/pmpp_matmul_v0.cu` | PMPP exercise: naive square matmul kernel + CPU verifier; `main` commented out. The kernel's inner loop starts at `i = row` rather than 0 (bug), and the commented `main` calls a non-existent `init_matrices`. | Not in the build. |
| `experiments/pmpp_matmul_v1.cu` | PMPP exercise: row-per-thread, column-per-thread, element-per-thread matmul and a mat-vec kernel with `cudaEvent` timing over 10 runs; `main` commented out. Verifier asserts exact fp32 equality against a CPU loop (works because inputs are small integers). | Not in the build. |
