#include <iostream>
#include <vector>
#include <functional>
#include <deque>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <cassert>



using Scalar = double;





struct Value {

    Value(double x) : data{x} {alive++;}
    Scalar data;
    Scalar grad {0.0};
    int pending {0};

    std::vector<Value*> prev;
    std::function<void()> backward;

    inline static int alive = 0;

    ~Value(){ alive--;}

};



class GraphContext {
public :
    std::vector<std::unique_ptr<Value>> nodes;

    Value* make(double data) {
        nodes.emplace_back(std::make_unique<Value>(data));
        return nodes.back().get();
    }
};





Value* mul(GraphContext& ctx, Value* a, Value* b) {
    Value* out = ctx.make(a->data * b->data);
    out->prev = {a, b};
    a->pending++;                       // out consumes a
    b->pending++;                       // out consumes b
    out->backward = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };
    return out;
}

Value* add(GraphContext& ctx, Value* a, Value* b) {
    Value* out = ctx.make(a->data + b->data);
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    out->backward = [a, b, out]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
}

Value* sub(GraphContext& ctx, Value* a, Value* b) {
    Value* out = ctx.make(a->data - b->data);
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    out->backward = [a, b, out]() {
        a->grad += out->grad;
        b->grad -= out->grad;
    };
    return out;
}

Value* exp(GraphContext& ctx, Value* a) {
    Value* out = ctx.make(std::exp(a->data) );
    out->prev = {a};
    a->pending++;
    out->backward = [a, out]() {
        a->grad += out->grad * out->data;
    };
    return out;
}

Value* div(GraphContext& ctx, Value* a, Value* b) {
    Value* out = ctx.make(a->data / b->data);
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    out->backward = [a, b, out]() {
        a->grad += out->grad * (1.0 / b->data);
        b->grad += out->grad * (-(a->data) / (b->data * b->data));
    };
    return out;
}

Value* log(GraphContext& ctx, Value* a) {
    Value* out = ctx.make(std::log(a->data));
    out->prev = {a};
    a->pending++;
    out->backward = [a, out]() {
        a->grad += out->grad * (1.0 / a->data);
    };
    return out;
}

Value* pow(GraphContext& ctx, Value* a, double raise) {
    Value* out = ctx.make(std::pow(a->data, raise) );
    out->prev = {a};
    a->pending++;
    out->backward = [a, raise, out]() {
        a->grad += out->grad * raise * std::pow(a->data, raise - 1);
    };
    return out;
}


Value* tanh(GraphContext& ctx, Value* a) {
    Value* out = ctx.make(std::tanh(a->data));
    out->prev = {a};
    a->pending++;
    out->backward = [a, out]() {                             // derivative of a tanh(x) operation is = 1 - tanh^2(x) ->
        a->grad += out->grad * (1 - (out->data * out->data));   //   = 1 - pow(out->data, 2)
    };
    return out;
}

Value* relu(GraphContext& ctx, Value* a) {
    Value* out = ctx.make(a->data > 0 ? a->data : 0.0);
    out->prev = {a};
    a->pending++;
    if (out->data) {           //derivative = 1 if a->data > 0, derivative is 0.0 if a->data <= 0
        out->backward = [a, out]() {
            a->grad += out->grad;
        };
        return out;
    }

    out->backward = [a, out]() {
        a->grad += 0;
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


bool gradcheck(std::function<Graph(GraphContext&, const std::vector<double>&)> build,
               std::vector<double> xs)
{

    //No h reaches 16 digits: shrinking h trades truncation for cancellation,
    //and the floor at their crossing is ≈ 3e-11 (~11 digits). h is tuned to the valley bottom, not to eps.

    const double h = 1e-5;                       // step size

    // ---- analytic side:
    GraphContext gc_ctx;
    Graph g = build(gc_ctx, xs);                         // BUILD CALL. Fresh nodes, pendings
    backwards(g.L);                              // fills every leaf's ->grad via chain rule

    bool all_ok = true;
    for (size_t i = 0; i < xs.size(); ++i) {
        // ---- numeric side for leaf i:
        std::vector<double> xph = xs;  xph[i] += h;
        std::vector<double> xmh = xs;  xmh[i] -= h;

        double f_plus;
        {
            GraphContext plus_ctx;
            f_plus  = build(plus_ctx, xph).L->data;
        }

        double f_minus;
        {
            GraphContext minus_ctx;
            f_minus  = build(minus_ctx, xmh).L->data;
        }
        double numeric  = (f_plus - f_minus) / (2 * h);
        double analytic = g.leaves[i]->grad;

        bool ok = compare_grad(analytic, numeric);
        all_ok = all_ok && ok;
        printf("leaf %zu:  analytic % .12f   numeric % .12f   %s\n",
               i, analytic, numeric, ok ? "PASS" : "FAIL");
    }
    return all_ok;
}




int main() {



    //Using Builders to test ops against graph configs:
    //Test inputs are chosen by the test author. Keep every intermediate O(1) so the ruler runs at its ~1e-11 floor,
    //far above the 1e-5 verdict line. The formula being certified is scale-free,
    //so certification at tame scale transfers to every scale.


    // Testing add(), mul(), sub(), and exp() in 'build'
    auto build = [](GraphContext& ctx, const std::vector<double>& xs) {
        Value* a_check = ctx.make(xs[0]);
        Value* b_check = ctx.make(xs[1]);
        Value* k_check = ctx.make(xs[2]);
        Value* m_check = ctx.make(xs[3]);
        Value* c_check = mul(ctx, a_check, b_check);
        Value* d_check = mul(ctx, c_check, k_check);
        Value* e_check = mul(ctx, c_check, m_check);
        Value* f_check = add(ctx, e_check, d_check);
        Value* j_exp = exp(ctx, f_check);
        Value* g_check = sub(ctx, j_exp, e_check);
        Value* pow_check = pow(ctx, g_check, 2.5);
        Value* h_check = mul(ctx, g_check, pow_check);
        Value* i_check = div(ctx, h_check, f_check);
        Value* n_check = log(ctx, i_check);
        Value* o_relu = relu(ctx, h_check);
        Value* L_check = mul(ctx, o_relu, n_check);
        return Graph{L_check, {a_check, b_check, k_check, m_check}};
    };

    gradcheck(build, {1.1, 0.7, 0.5, 0.3});

    assert(Value::alive == 0);

    // Testing leaf -> exp() -> out in 'build2'
    auto build2 = [](GraphContext& ctx2,  const std::vector<double>& xs) {
        Value* a2_check = ctx2.make(xs[0]);
        Value* L2_check = exp(ctx2, a2_check);
        return Graph{L2_check, {a2_check}};
    };

    gradcheck(build2, {1.5});

    assert(Value::alive == 0);

    // Test leaf -> tanh() -> out in build3
    auto build3 = [](GraphContext& ctx3, const std::vector<double>& xs) {
        Value* a3_check = ctx3.make(xs[0]);
        Value* L3_check = tanh(ctx3, a3_check);
        return Graph{L3_check, {a3_check}};
    };

    gradcheck(build3, {1.5});

    assert(Value::alive == 0);

    auto build4 = [](GraphContext& ctx4, const std::vector<double>& xs) {
        Value* a4_check = ctx4.make(xs[0]);
        Value* L4_check = relu(ctx4, a4_check);
        return Graph{L4_check, {a4_check}};
    };

    // test is inaccurate where x=0 & grad = 0
    gradcheck(build4, {1.5});

    assert(Value::alive == 0);

    gradcheck(build4, {-1.5});

    assert(Value::alive == 0);

    return 0;
}

