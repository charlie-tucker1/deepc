#pragma once
#include "deepc/tensor.h"



namespace deepc {

    Tensor* add(tensorGraphContext& ctx, Tensor* a, Tensor* b);

    Tensor* bias_add(tensorGraphContext& ctx, Tensor* a, Tensor* bias);

    Tensor* mul(tensorGraphContext& ctx, Tensor* a, Tensor* b);

    Tensor* relu(tensorGraphContext& ctx, Tensor* a);

    Tensor* cross_entropy_loss(tensorGraphContext& ctx, Tensor* logits, const std::vector<int>& labels);


}