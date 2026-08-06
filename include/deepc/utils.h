#pragma once


namespace deepc {

    struct Tensor;
    class tensorGraphContext;

    enum class Backend {CPU, CUDA}; // used by tensorGraphContext to determine backend (behind ifdef DEEPC_CUDA)

    Tensor* clone(tensorGraphContext& ctx, const Tensor* src);

    void sgd_step(tensorGraphContext& ctx, double lr);

    void zero_grad(tensorGraphContext& params);


    void backwards(Tensor* loss);


}