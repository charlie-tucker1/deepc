#include <deque>
#include <vector>
#include "deepc/tensor.h"
#include "deepc/utils.h"



namespace deepc {

    class GraphArena;

    Tensor* clone(GraphArena& arena, const Tensor* src) {
        Tensor* t = arena.make(src->rows, src->cols);
        std::copy(src->data.get(), src->data.get() + src->rows*src->cols, t->data.get());
        return t;   // caller reassigns their own variable
    }


    void sgd_step(GraphArena& params, float lr) {
        for (auto& up : params.tensors) {          // up: unique_ptr<Tensor>&
            Tensor* t = up.get();
            int n = t->rows * t->cols;
            for (int i = 0; i < n; i++)
                t->data[i] -= lr * t->grad[i];
        }
    }

    void zero_grad(GraphArena& params) {
        for (auto& up : params.tensors) {          // up: unique_ptr<Tensor>&
            Tensor* t = up.get();
            int n = t->rows * t->cols;
            std::fill(t->grad.get(), t->grad.get() + n, 0.0f);
        }
    }



    void backwards(Tensor* loss) {
        int n = loss->rows * loss->cols;
        for (int i {0}; i < n; i++) {loss->grad[i] = 1.0;}

        std::deque<Tensor*> ready {loss};
        while (!(ready.empty())) {
            Tensor* t = ready.front(); ready.pop_front();
            if (t->backward) t->backward();
            for (Tensor* v : t->prev) {
                v->pending--;
                if (v->pending == 0) ready.emplace_back(v);
            }
        }
    }



}
