#include <iostream>
#include <vector>
#include <functional>
#include <deque>
#include <cmath>
#include <algorithm>
#include <iostream>



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

Value* add( Value* a, Value* b) {
    auto out = new Value{a->data + b->data};
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    out->backward = [a, b, out]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
}

Value* sub( Value* a, Value* b) {
    auto out = new Value{a->data - b->data};
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    out->backward = [a, b, out]() {
        a->grad += out->grad;
        b->grad -= out->grad;
    };
    return out;
}

Value* exp(Value* a) {
    auto out = new Value{std::exp(a->data) };
    out->prev = {a};
    a->pending++;
    out->backward = [a, out]() {
        a->grad += out->grad * out->data;
    };
    return out;
}

Value* div(Value* a, Value* b) {
    auto out = new Value{a->data / b->data};
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    out->backward = [a, b, out]() {
        a->grad += out->grad * (1.0 / b->data);
        b->grad += out->grad * (-(a->data) / (b->data * b->data));
    };
    return out;
}

Value* log(Value* a) {
    auto out = new Value{std::log(a->data) };
    out->prev = {a};
    a->pending++;
    out->backward = [a, out]() {
        a->grad += out->grad * (1.0 / a->data);
    };
    return out;
}

Value* pow(Value* a, double raise) {
    auto out = new Value{std::pow(a->data, raise) };
    out->prev = {a};
    a->pending++;
    out->backward = [a, raise, out]() {
        a->grad += out->grad * raise * std::pow(a->data, raise - 1);
    };
    return out;
}


Value* tanh(Value* a) {
    Value* out = new Value{std::tanh(a->data) };
    out->prev = {a};
    a->pending++;
    out->backward = [a, out]() {                             // derivative of a tanh(x) operation is = 1 - tanh^2(x) ->
        a->grad += out->grad * (1 - (out->data * out->data));   //   = 1 - pow(out->data, 2)
    };
    return out;
}

Value* relu(Value* a) {
    Value* out = new Value;
    out->prev = {a};
    a->pending++;
    if (a->data > 0) {           //derivative = 1 if a->data > 0, derivative is 0.0 if a->data <= 0
        out->data = a->data;
        out->backward = [a, out]() {
            a->grad += out->grad;
        };
        return out;
    }
    else {
        out->data = 0.0;
        out->backward = [a, out]() {
            a->grad += 0;
        };
        return out;
    }
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

    //No h reaches 16 digits: shrinking h trades truncation for cancellation,
    //and the floor at their crossing is ≈ 3e-11 (~11 digits). h is tuned to the valley bottom, not to eps.

    const double h = 1e-5;                       // step size

    // ---- analytic side:
    Graph g = build(xs);                         // BUILD CALL. Fresh nodes, pendings
    backwards(g.L);                              // fills every leaf's ->grad via chain rule

    bool all_ok = true;
    for (size_t i = 0; i < xs.size(); ++i) {
        // ---- numeric side for leaf i:
        std::vector<double> xph = xs;  xph[i] += h;
        std::vector<double> xmh = xs;  xmh[i] -= h;
        double f_plus  = build(xph).L->data;
        double f_minus = build(xmh).L->data;
        double numeric  = (f_plus - f_minus) / (2 * h);
        double analytic = g.leaves[i]->grad;

        bool ok = compare_grad(analytic, numeric);
        all_ok = all_ok && ok;
        printf("leaf %zu:  analytic % .12f   numeric % .12f   %s\n",
               i, analytic, numeric, ok ? "PASS" : "FAIL");
    }
    return all_ok;    // TODO: every build() leaks its graph, fix before moving to tensors
}

int main() {



    //Using Builders to test ops against graph configs:
    //Test inputs are chosen by the test author. Keep every intermediate O(1) so the ruler runs at its ~1e-11 floor,
    //far above the 1e-5 verdict line. The formula being certified is scale-free,
    //so certification at tame scale transfers to every scale.


    // Testing add(), mul(), sub(), and exp() in 'build'
    auto build = [](const std::vector<double>& xs) {
        Value* a_check = new Value{xs[0]};
        Value* b_check = new Value{xs[1]};
        Value* k_check = new Value{xs[2]};
        Value* m_check = new Value{xs[3]};
        Value* c_check = mul(a_check, b_check);
        Value* d_check = mul(c_check, k_check);
        Value* e_check = mul(c_check, m_check);
        Value* f_check = add(e_check, d_check);
        Value* j_exp = exp(f_check);
        Value* g_check = sub(j_exp, e_check);
        Value* pow_check = pow(g_check, 2.5);
        Value* h_check = mul(g_check, pow_check);
        Value* i_check = div(h_check, f_check);
        Value* n_check = log(i_check);
        Value* o_relu = relu(h_check);
        Value* L_check = mul(o_relu, n_check);
        return Graph{L_check, {a_check, b_check, k_check, m_check}};
    };

    gradcheck(build, {1.1, 0.7, 0.5, 0.3});


    // Testing leaf -> exp() -> out in 'build2'
    auto build2 = [](const std::vector<double>& xs) {
        Value* a2_check = new Value{xs[0]};
        Value* L2_check = exp(a2_check);
        return Graph{L2_check, {a2_check}};
    };

    gradcheck(build2, {1.5});


    // Test leaf -> tanh() -> out in build3
    auto build3 = [](const std::vector<double>& xs) {
        Value* a3_check = new Value{xs[0]};
        Value* L3_check = tanh(a3_check);
        return Graph{L3_check, {a3_check}};
    };

    gradcheck(build3, {1.5});

    auto build4 = [](const std::vector<double>& xs) {
        Value* a4_check = new Value{xs[0]};
        Value* L4_check = relu(a4_check);
        return Graph{L4_check, {a4_check}};
    };

    // test is inaccurate where x=0 & grad = 0
    gradcheck(build4, {1.5});
    gradcheck(build4, {-1.5});

    return 0;
}
