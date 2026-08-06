#include "deepc/tensor.h"
#include "deepc/utils.h"
#include "deepc/gradcheck.h"

#include <cmath>
#include <random>
#include <vector>


namespace deepc {
    class GraphArena;
    struct Tensor;

    bool compare_grad_t(const double a, const double n) {
        const double atol = 1e-8;
        const double rtol = 1e-5;
        return std::abs(a - n) < atol + rtol * std::max(std::abs(a), std::abs(n));
    }

    bool tensor_gradcheck(std::function<Graph(GraphArena& arena, const std::vector<Tensor*>&)> build,
                   std::vector<Tensor*> leaves, int num_tests)
    {

        //No h reaches 16 digits: shrinking h trades truncation for cancellation,
        //and the floor at their crossing is ≈ 3e-11 (~11 digits). h is tuned to the valley bottom, not to eps.

        const double h = 1e-5;                           // step size

        // ---- analytic side:
        GraphArena an_arena;
        Graph g = build(an_arena, leaves);              // BUILD CALL. Fresh nodes, pendings
        backwards(g.L);                                  // fills every leaf's ->grad via chain rule

        std::random_device rd;
        std::mt19937 gen(rd());

        bool all_ok = true;
        for (size_t i = 0; i < num_tests; ++i) {


            std::uniform_int_distribution<int> dist_leaves(0,leaves.size() - 1);
            int nudgeLeaf = dist_leaves(gen);
            std::uniform_int_distribution<int> dist_elements(0,(leaves[nudgeLeaf]->rows * leaves[nudgeLeaf]->cols) - 1);
            int nudgeIdx = dist_elements(gen);

            double f_plus {0.0};
            {
                GraphArena plus_arena;
                std::vector<Tensor*> plus_leaves;
                for (Tensor* x : leaves) plus_leaves.emplace_back(clone(plus_arena, x));

                plus_leaves[nudgeLeaf]->data[nudgeIdx] += h;      // mutate the CLONE, xs untouched

                Graph fpg = build(plus_arena, plus_leaves);

                for (int i = 0; i < fpg.L->rows * fpg.L->cols; i++) f_plus  += fpg.L->data[i];
            }

            double f_minus {0.0};
            {
                GraphArena minus_arena;
                std::vector<Tensor*> minus_leaves;
                for (Tensor* x : leaves) minus_leaves.emplace_back(clone(minus_arena, x));

                minus_leaves[nudgeLeaf]->data[nudgeIdx] -= h;

                Graph fmg = build(minus_arena, minus_leaves);

                for (int i = 0; i < fmg.L->rows * fmg.L->cols; i++) f_minus  += fmg.L->data[i];
            }


            double numeric  = (f_plus - f_minus) / (2 * h);
            double analytic = g.leaves[nudgeLeaf]->grad[nudgeIdx];

            bool ok = compare_grad_t(analytic, numeric);
            all_ok = all_ok && ok;
            printf("leaf: %d , elem %d:  analytic % .12f   numeric % .12f   %s\n",
                   nudgeLeaf, nudgeIdx, analytic, numeric, ok ? "PASS" : "FAIL");
        }
        return all_ok;
    }
}
