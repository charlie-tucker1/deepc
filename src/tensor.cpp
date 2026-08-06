#include "deepc/tensor.h"

#include <memory>
#include <random>

namespace deepc {

    Tensor::Tensor(int rows, int cols)
        : rows{rows}, cols{cols},
          data{std::make_unique<double[]>(rows * cols)},
          grad{std::make_unique<double[]>(rows * cols)}
    {
        alive++;
    }

    Tensor::~Tensor() {
        alive--;
    }

    void Tensor::init_tensor_random(float minBound, float maxBound) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(minBound, maxBound);
        for (int i = 0; i < rows * cols; i++) data[i] = dist(gen);
    }

    Tensor* GraphArena::make(int rows, int cols) {
        tensors.emplace_back(std::make_unique<Tensor>(rows, cols));
        return tensors.back().get();
    }

}






