#include <iostream>
#include <vector>
#include <functional>
#include <deque>



using Scalar = double;


struct Value {

    Scalar data {0.0};
    Scalar grad {0.0};
    int pending {0};

    std::vector<Value*> prev;
    std::function<void()> backward;
};



Value* leaf(double x) {
    Value* v = new Value{x};
    return v;
}

Value* mul(Value* a, Value* b) {
    auto out = new Value{a->data * b->data};
    out->prev = {a, b};
    a->pending++;                       // out consumes a
    b->pending++;                       // out consumes b
    out->backward = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };
    return out;
}



void backwards(Value* L) {
    L->grad = 1.0;
    std::deque<Value*> ready = {L};
    while (!ready.empty()) {
        Value* node = ready.front(); ready.pop_front();
        if (node->backward) node->backward();
        for (Value* p : node->prev) {
            p->pending--;
            if (p->pending == 0) ready.push_back(p);
        }
    }
}


struct Graph {
    Value* L;
    std::vector<Value*> leaves;
};

bool compare_grad(double a, double n) {
    const double atol = 1e-8;
    const double rtol = 1e-5;
    return std::abs(a - n) < atol + rtol * std::max(std::abs(a), std::abs(n));
}


bool gradcheck(std::function<Graph(const std::vector<double>&)> build,
               std::vector<double> xs)
{
    const double h = 1e-5;                       // error floor

    // ---- analytic side: fresh graph #1, then let the engine do its thing ----
    Graph g = build(xs);                         // BUILD CALL. Fresh nodes, virgin pendings.
    backwards(g.L);                               // fills every leaf's ->grad via chain rule

    bool all_ok = true;
    for (size_t i = 0; i < xs.size(); ++i) {
        // ---- numeric side for leaf i: two MORE fresh graphs, forward-only ----
        std::vector<double> xph = xs;  xph[i] += h;
        std::vector<double> xmh = xs;  xmh[i] -= h;
        double f_plus  = build(xph).L->data;      // BUILD CALL. backward never runs here.
        double f_minus = build(xmh).L->data;      // BUILD CALL. We only want the number.
        double numeric  = (f_plus - f_minus) / (2 * h);   // parens on BOTH groups
        double analytic = g.leaves[i]->grad;     // calculus's answer, from graph #1

        bool ok = compare_grad(analytic, numeric);          // NUMBERS compared. Not graphs.
        all_ok = all_ok && ok;
        printf("leaf %zu:  analytic % .12f   numeric % .12f   %s\n",
               i, analytic, numeric, ok ? "PASS" : "FAIL");
    }
    return all_ok;    // TODO: every build() leaks its graph, fix before moving to tensors
}

int main() {
    Value* a = leaf(2.0);
    Value* b = leaf(3.0);
    Value* k = leaf(4.0);
    Value* m = leaf(5.0);
    Value* c = mul(a, b);
    Value* d = mul(c, k);
    Value* e = mul(c, m);
    Value* L = mul(d, e);

    backwards(L);

    auto build = [](const std::vector<double>& xs) {
        Value* a_check = new Value{xs[0]};
        Value* b_check = new Value{xs[1]};
        Value* k_check = new Value{xs[2]};
        Value* m_check = new Value{xs[3]};
        Value* c_check = mul(a_check, b_check);
        Value* d_check = mul(c_check, k_check);
        Value* e_check = mul(c_check, m_check);
        Value* L_check = mul(d_check, e_check);
        return Graph{L_check, {a_check, b_check, k_check, m_check}};
    };

    bool ok = gradcheck(build, {2.0, 3.0, 4.0, 5.0});


    return ok;
}
