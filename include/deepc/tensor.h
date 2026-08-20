#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "utils.h"

namespace deepc {
    struct Tensor {
        inline static int alive {0}; // for debugging / detecting mem leaks

        Tensor(int rows, int cols);
        int rows, cols;
        std::unique_ptr<float[]> data;
        std::unique_ptr<float[]> grad;

        int pending{0};

        std::vector<Tensor*> prev;
        std::function<void()> backward;

        void init_tensor_random(float minBound, float maxBound);

        ~Tensor();
    };

    class GraphArena {
    public:
        std::vector<std::unique_ptr<Tensor>> tensors; //unique pointers manage graph memory lifespan/ownership

        Backend backend = Backend::CPU;

        Tensor* make(int rows, int cols);
    };
}
