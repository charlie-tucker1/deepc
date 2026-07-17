#include <iostream>
#include <vector>
#include <functional>
#include <deque>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <cassert>





struct Tensor {

    Tensor(int rows, int cols) : rows{rows}, cols{cols},
        data{std::make_unique<double[]>(rows * cols)},
        grad{std::make_unique<double[]>(rows * cols)}
    {
        alive++;
    }


    inline static int alive {0};
    int rows, cols;
    std::unique_ptr<double[]> data;
    std::unique_ptr<double[]> grad;


    int pending {0};

    std::vector<Tensor*> prev;

    std::function<void()> backward;

    ~Tensor() {
        alive--;
    }


};

class T_GraphContext {
public:
    std::vector<std::unique_ptr<Tensor>> tensors;

    Tensor* make(int rows, int cols) {
        tensors.emplace_back(std::make_unique<Tensor>(rows, cols));
        return tensors.back().get();
    }
};


// TODO: write elwise add and begin Tensor autograd engine!!!

void elwise_add(const double *a, const double *b, double *out, int n) {


}