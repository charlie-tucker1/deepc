#pragma once
#include "deepc/tensor.h"



namespace deepc {

    Tensor* add(GraphArena& arena, Tensor* a, Tensor* b);

    Tensor* bias_add(GraphArena& arena, Tensor* a, Tensor* bias);

    Tensor* mul(GraphArena& arena, Tensor* a, Tensor* b);

    Tensor* relu(GraphArena& arena, Tensor* a);

    Tensor* cross_entropy_loss(GraphArena& arena, Tensor* logits, const std::vector<int>& labels);


}