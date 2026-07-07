#include <iostream>
#include <vector>
#include <functional>



using Scalar = float64_t;


struct Value {

    Scalar data;
    Scalar grad {0.0};
    int pending = 0;

    std::vector<Value*> prev;
    std::function<void()> backward;
};

std::vector<Value*> tape;

Value* leaf(double x) {
    Value* v = new Value{x};
    tape.push_back(v);
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
    tape.push_back(out);
    return out;
}


void backwards(Value* L, const std::vector<Value*>& tape) {
    L->grad = 1.0;
    std::vector<Value*> ready;
    for (Value * v : tape) {
        if (!(v->pending)) {
        ready.push_back(v);}
    }
    while (!ready.empty()) {
        Value* t = ready.back();
        ready.pop_back();

        if (t->backward) t->backward();

        for (Value* p : t->prev) {
            p->pending--;
            if (!p->pending) {
                ready.push_back(p);
            }
        }

    }

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

    backwards(L, tape);

    printf("a %g  b %g  k %g  m %g\nc %g  d %g  e %g  L %g\n",
           a->grad, b->grad, k->grad, m->grad, c->grad, d->grad, e->grad, L->grad);
    return 0;
}