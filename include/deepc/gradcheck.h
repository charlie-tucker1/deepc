#pragma once

#include <vector>
#include <functional>

namespace deepc {
    struct Tensor;
    class tensorGraphContext;

    struct Graph {
        Tensor* L;
        std::vector<Tensor*> leaves;
    };

    bool compare_grad_t(double a, double b);

    bool tensor_gradcheck(std::function<Graph(tensorGraphContext& ctx, const std::vector<Tensor*>&)> build,
                        std::vector<Tensor*> leaves, int num_tests);




}
