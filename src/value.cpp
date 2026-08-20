#include <iostream>
#include <vector>
#include <functional>
#include <deque>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <cassert>



using Scalar = float;





struct value {

    value(float x) : data{x} {alive++;}
    Scalar data;
    Scalar grad {0.0};
    int pending {0};

    std::vector<value*> prev;
    std::function<void()> backward;

    inline static int alive = 0;

    ~value(){ alive--;}

};



class GraphContext {
public :
    std::vector<std::unique_ptr<value>> nodes;

    value* make(float data) {
        nodes.emplace_back(std::make_unique<value>(data));
        return nodes.back().get();
    }
};





value* mul(GraphContext& arena, value* a, value* b) {
    value* out = arena.make(a->data * b->data);
    out->prev = {a, b};
    a->pending++;                       // out consumes a
    b->pending++;                       // out consumes b
    out->backward = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };
    return out;
}

value* add(GraphContext& arena, value* a, value* b) {
    value* out = arena.make(a->data + b->data);
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    out->backward = [a, b, out]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
}

value* sub(GraphContext& arena, value* a, value* b) {
    value* out = arena.make(a->data - b->data);
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    out->backward = [a, b, out]() {
        a->grad += out->grad;
        b->grad -= out->grad;
    };
    return out;
}

value* exp(GraphContext& arena, value* a) {
    value* out = arena.make(std::exp(a->data) );
    out->prev = {a};
    a->pending++;
    out->backward = [a, out]() {
        a->grad += out->grad * out->data;
    };
    return out;
}

value* div(GraphContext& arena, value* a, value* b) {
    value* out = arena.make(a->data / b->data);
    out->prev = {a, b};
    a->pending++;
    b->pending++;
    out->backward = [a, b, out]() {
        a->grad += out->grad * (1.0 / b->data);
        b->grad += out->grad * (-(a->data) / (b->data * b->data));
    };
    return out;
}

value* log(GraphContext& arena, value* a) {
    value* out = arena.make(std::log(a->data));
    out->prev = {a};
    a->pending++;
    out->backward = [a, out]() {
        a->grad += out->grad * (1.0 / a->data);
    };
    return out;
}

value* pow(GraphContext& arena, value* a, float raise) {
    value* out = arena.make(std::pow(a->data, raise) );
    out->prev = {a};
    a->pending++;
    out->backward = [a, raise, out]() {
        a->grad += out->grad * raise * std::pow(a->data, raise - 1);
    };
    return out;
}


value* tanh(GraphContext& arena, value* a) {
    value* out = arena.make(std::tanh(a->data));
    out->prev = {a};
    a->pending++;
    out->backward = [a, out]() {                             // derivative of a tanh(x) operation is = 1 - tanh^2(x) ->
        a->grad += out->grad * (1 - (out->data * out->data));   //   = 1 - pow(out->data, 2)
    };
    return out;
}

value* relu(GraphContext& arena, value* a) {
    value* out = arena.make(a->data > 0 ? a->data : 0.0);
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



void backwards(value* L) {
    L->grad = 1.0;
    std::deque<value*> ready = {L};
    while (!ready.empty()) {
        value* node = ready.front(); ready.pop_front();
        if (node->backward) node->backward();
        for (value* p : node->prev) {
            p->pending--;
            if (p->pending == 0) ready.push_back(p);
        }
    }
}



struct Graph {
    value* L;
    std::vector<value*> leaves;
};

bool compare_grad(float a, float n) {
    const float atol = 1e-8;
    const float rtol = 1e-5;
    return std::abs(a - n) < atol + rtol * std::max(std::abs(a), std::abs(n));
}


bool gradcheck(std::function<Graph(GraphContext&, const std::vector<float>&)> build,
               std::vector<float> xs)
{

    //No h reaches 16 digits: shrinking h trades truncation for cancellation,
    //and the floor at their crossing is ≈ 3e-11 (~11 digits). h is tuned to the valley bottom, not to eps.

    const float h = 1e-5;                       // step size

    // ---- analytic side:
    GraphContext gc_arena;
    Graph g = build(gc_arena, xs);                         // BUILD CALL. Fresh nodes, pendings
    backwards(g.L);                              // fills every leaf's ->grad via chain rule

    bool all_ok = true;
    for (size_t i = 0; i < xs.size(); ++i) {
        // ---- numeric side for leaf i:
        std::vector<float> xph = xs;  xph[i] += h;
        std::vector<float> xmh = xs;  xmh[i] -= h;

        float f_plus;
        {
            GraphContext plus_arena;
            f_plus  = build(plus_arena, xph).L->data;
        }

        float f_minus;
        {
            GraphContext minus_arena;
            f_minus  = build(minus_arena, xmh).L->data;
        }
        float numeric  = (f_plus - f_minus) / (2 * h);
        float analytic = g.leaves[i]->grad;

        bool ok = compare_grad(analytic, numeric);
        all_ok = all_ok && ok;
        printf("leaf %zu:  analytic % .12f   numeric % .12f   %s\n",
               i, analytic, numeric, ok ? "PASS" : "FAIL");
    }
    return all_ok;
}



/*
int main() {



    //Using Builders to test ops against graph configs:
    //Test inputs are chosen by the test author. Keep every intermediate O(1) so the ruler runs at its ~1e-11 floor,
    //far above the 1e-5 verdict line. The formula being certified is scale-free,
    //so certification at tame scale transfers to every scale.


    // Testing add(), mul(), sub(), and exp() in 'build'
    auto build = [](GraphContext& arena, const std::vector<float>& xs) {
        Value* a_check = arena.make(xs[0]);
        Value* b_check = arena.make(xs[1]);
        Value* k_check = arena.make(xs[2]);
        Value* m_check = arena.make(xs[3]);
        Value* c_check = mul(arena, a_check, b_check);
        Value* d_check = mul(arena, c_check, k_check);
        Value* e_check = mul(arena, c_check, m_check);
        Value* f_check = add(arena, e_check, d_check);
        Value* j_exp = exp(arena, f_check);
        Value* g_check = sub(arena, j_exp, e_check);
        Value* pow_check = pow(arena, g_check, 2.5);
        Value* h_check = mul(arena, g_check, pow_check);
        Value* i_check = div(arena, h_check, f_check);
        Value* n_check = log(arena, i_check);
        Value* o_relu = relu(arena, h_check);
        Value* L_check = mul(arena, o_relu, n_check);
        return Graph{L_check, {a_check, b_check, k_check, m_check}};
    };

    gradcheck(build, {1.1, 0.7, 0.5, 0.3});

    assert(Value::alive == 0);

    // Testing leaf -> exp() -> out in 'build2'
    auto build2 = [](GraphContext& arena2,  const std::vector<float>& xs) {
        Value* a2_check = arena2.make(xs[0]);
        Value* L2_check = exp(arena2, a2_check);
        return Graph{L2_check, {a2_check}};
    };

    gradcheck(build2, {1.5});

    assert(Value::alive == 0);

    // Test leaf -> tanh() -> out in build3
    auto build3 = [](GraphContext& arena3, const std::vector<float>& xs) {
        Value* a3_check = arena3.make(xs[0]);
        Value* L3_check = tanh(arena3, a3_check);
        return Graph{L3_check, {a3_check}};
    };

    gradcheck(build3, {1.5});

    assert(Value::alive == 0);

    auto build4 = [](GraphContext& arena4, const std::vector<float>& xs) {
        Value* a4_check = arena4.make(xs[0]);
        Value* L4_check = relu(arena4, a4_check);
        return Graph{L4_check, {a4_check}};
    };

    // test is inaccurate where x=0 & grad = 0
    gradcheck(build4, {1.5});

    assert(Value::alive == 0);

    gradcheck(build4, {-1.5});

    assert(Value::alive == 0);

    return 0;
}
*/
