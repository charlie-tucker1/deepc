#pragma once

#include <vector>
#include <functional>

namespace deepc {
    struct Tensor;
    class GraphArena;

    struct Graph {
        Tensor* L;
        std::vector<Tensor*> leaves;
    };

    bool compare_grad_t(double a, double b);

    bool tensor_gradcheck(std::function<Graph(GraphArena& arena, const std::vector<Tensor*>&)> build,
                        std::vector<Tensor*> leaves, int num_tests);




}
