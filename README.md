# deepC

A from-scratch neural network training engine in C++. No frameworks, no BLAS, no dependencies beyond the standard library. Every op, gradient, and memory-ownership decision is handwritten and verified by numerical gradient checking.

Current status: **MNIST MLP trains end-to-end on CPU to 95%+ test accuracy.** Next phase: CUDA backend targeting an RTX 5070.

The point of the project is not MNIST. The point is building all the fundamental deep learning machinery by hand: tensor storage, computational graph construction, reverse-mode autodiff, optimizer, data pipeline, and gradcheck. Then moving the computational layer to CUDA.

---

## Results

MLP 784 → 128 (ReLU) → 10, softmax cross-entropy, SGD. 60k train / 10k test MNIST.

| Build   | Batch | lr   | Epochs | Train / epoch | Inference (10k) | Test acc |
|---------|-------|------|--------|---------------|-----------------|----------|
| Debug   | 1     | 0.01 | 1      | 203 s         | 4.3 s           | 95.85%   |
| Debug   | 64    | 0.01 | 1      | 138 s         | 4.3 s           | 86.7%    |
| Release | 64    | 0.01 | 1      | 56.7 s        | 0.46 s          | 86.25%   |
| Release | 64    | 0.05 | 5      | ~57 s         | 0.50 s          | 95.22%   |
| Release | 1     | 0.05 | 5      | 44.6 s        | 0.54 s          | 94.05%   |

Fixed batch order (no per-epoch shuffle) in all runs.

### Findings the table encodes

**1. Debug→Release speedup is wildly asymmetric: 9.4× for inference, 2.4× for training.**
The forward path (ikj-ordered matmul, streaming writes, no loop-carried dependencies) vectorizes cleanly under `-O2`. The backward pass does not: both weight-gradient loops accumulate into fixed memory locations through raw pointers the compiler cannot prove non-aliasing, forcing serial memory-round-trip reductions. The bottleneck is loop *structure*, not FLOPs.

**2. Batching inverts sign between build modes.** B=1 → B=64 makes Debug training 1.47× *faster* (amortizes per-step overhead: graph construction, allocation, and full-parameter `sgd_step`/`zero_grad` sweeps — 60000 vs 937 per epoch) but makes Release training 1.27× *slower* (the batch dimension adds a large-stride walk to the weight-gradient inner loop that the optimizer can't save). Batching helps when overhead dominates and hurts when loop quality dominates. The batch dimension the CPU backward leaves on the table is exactly the axis a GPU consumes — this is the empirical case for the CUDA phase.

**3. The lr × batch-size coupling, measured from both directions.** At B=64, raising lr 0.01 → 0.05 is what unlocked 95% (batch-mean gradients are low-variance; step size was the binding constraint). The same lr=0.05 at B=1 lands *below* the one-epoch lr=0.01 baseline despite 5× the optimization — single-example gradients are 64× noisier and the noise floor binds instead.

---

## Architecture

### Ownership: arena contexts, two lifetimes

All tensors are owned by a `tensorGrapharena` — an arena of `unique_ptr<Tensor>` — and referenced everywhere else by raw pointer. Two context lifetimes:

- **Persistent** — model parameters live in the `MNIST_MLP`'s own `params` arena for the life of the model.
- **Per-step** — each training step builds its graph (input tensor, activations, loss) in a fresh `step_arena` that dies at the bottom of the loop iteration. Graph teardown is scope exit; there is no `free` logic to get wrong.

A `Tensor::alive` counter exists purely to catch leaks during development.

### Ops and autodiff

Ops are free functions (`mul`, `add`, `bias_add`, `relu`, `cross_entropy_loss`) that:

1. allocate their output in the caller's arena,
2. record graph edges (`out->prev`, `input->pending++`),
3. compute the forward immediately (eager),
4. capture a backward closure in `out->backward` holding whatever forward state the gradient needs.

`backwards(loss)` seeds `loss->grad = 1` and runs a pending-count topological traversal: a node's closure fires only when all its consumers have fired, so gradient accumulation via `+=` is always complete before propagating further.

Cross-entropy is a fused graph op — row-wise max-subtracted softmax, mean of per-row NLL, with per-row `m`/`psum` stashed in the closure by value. Mean (not sum) over the batch keeps lr semantics independent of B. The `labels.size() == logits->rows` assert makes batch size a one-variable knob.

### Gradient checking

`tensor_gradcheck` verifies every op and the full composed MLP against central finite differences (`h = 1e-5`, tolerance `atol 1e-8 + rtol 1e-5`). Functionality:

- **The build lambda must use the passed-in `leaves` directly.** The harness owns cloning for the ±h numeric passes; the analytic pass must wire the real leaves into the graph so `backwards` fills the grads the harness reads. No cloning required in build (symptom: exact-zero analytic gradients).
- **The build lambda must be deterministic in its leaves.** Same inputs → same graph → same loss. Fresh randomness inside `build` means f(+h) and f(−h) are evaluated on *different functions* and the numeric column is garbage. Non-leaf inputs (e.g. the data tensor) are hoisted out and cloned in.

Every op in the engine is certified at batch dimensions, both raw and composed through the full network.

---

## Building & running

Standard CMake project (developed in CLion on Windows; also builds on Linux).

```
cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
./cmake-build-release/deepc
```

Expects the four MNIST idx files under `data/MNIST_data/`:

```
train-images.idx3-ubyte    train-labels.idx1-ubyte
t10k-images.idx3-ubyte     t10k-labels.idx1-ubyte
```

`DEEPC_DATA_DIR` is set by CMake to the repo's `data/` directory and falls back to a relative `data/` when run from the repo root. Both Debug and Release profiles are used deliberately, the Debug/Release delta is part of the measurement.

---

## Roadmap

- **CUDA backend** — port forward/backward matmul to the GPU (RTX 5070), batch dimension mapped to the grid. The CPU table above is the baseline.
- Per-phase segment timing (forward / backward / optimizer) to decompose the training-time budget.
- Per-epoch shuffle in the training loop.
- Batched inference path (current eval runs B=1; correctness-identical, just slower than it needs to be).
