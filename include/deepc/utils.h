#pragma once


namespace deepc {

    struct Tensor;
    class GraphArena;

    enum class Backend {CPU, CUDA}; // used by GraphArena to determine backend (behind ifdef DEEPC_CUDA)

    Tensor* clone(GraphArena& arena, const Tensor* src);

    void sgd_step(GraphArena& arena, double lr);

    void zero_grad(GraphArena& params);


    void backwards(Tensor* loss);


}