
// Set by CMake to <repo>/data; falls back to running from the repo root
#ifndef DEEPC_DATA_DIR
#define DEEPC_DATA_DIR "data"
#endif

#include <algorithm>
#include <iostream>
#include <chrono>
#include "deepc/tensor.h"
#include "deepc/ops.h"
#include "deepc/mnist_loader.h"

namespace deepc {
    class tensorGraphContext;

    struct MNIST_MLP {
        tensorGraphContext params;        // owns permanent params
        Tensor *W1,  *W2, *b1, *b2;    // raw ptrs into params

        MNIST_MLP(const int in, const int hidden, const int out) {

            W1 = params.make(in, hidden);  b1 = params.make(1, hidden);
            W2 = params.make(hidden, out); b2 = params.make(1, out);

        }

        // the architecture for our MNIST MLP:
        static Tensor* mnist_forward_ops(tensorGraphContext& ctx, Tensor* x,
                                   Tensor* W1, Tensor* b1, Tensor* W2, Tensor* b2) {
            Tensor* h = relu(ctx, bias_add(ctx, mul(ctx, x, W1), b1));
            return bias_add(ctx, mul(ctx, h, W2), b2);
        }

        Tensor* forward(tensorGraphContext& ctx, Tensor* x) {
            return mnist_forward_ops(ctx, x, W1, b1, W2, b2);
        }


        int infer(Tensor* x) {
            tensorGraphContext ctx;
            Tensor* logits = this->forward(ctx, x);
            for (const auto& up : params.tensors) up->pending = 0;   // forward-without-backwards reset, as designed
            return std::distance(&logits->data[0],
                std::max_element(&logits->data[0], &logits->data[0]+10));
        }
    };
}


    int main() {

        using namespace deepc;

        // Load MNIST dataset:

        const MNIST mnist = load_mnist(DEEPC_DATA_DIR "/MNIST_data/train-images.idx3-ubyte",
                                                       DEEPC_DATA_DIR "/MNIST_data/train-labels.idx1-ubyte");

        std::cout << "Loaded " << mnist.images.size() << " images\n";
        std::cout << "Rows: " << mnist.rows << ", Cols: " << mnist.cols << "\n";
        std::cout << "First label: " << static_cast<int>(mnist.labels[0]) << "\n";
        std::cout << "First pixel: " << static_cast<int>(mnist.images[0][0]) << "\n";



        //Init our MLP with proper dims, randomly init by: +- 1 / sqrt(->dim)
        MNIST_MLP mnist_model(784, 128, 10);
        mnist_model.W1->init_tensor_random(-0.04, 0.04);
        mnist_model.b1->init_tensor_random(-0.04, 0.04);
        mnist_model.W2->init_tensor_random(-0.09, 0.09);
        mnist_model.b2->init_tensor_random(-0.09, 0.09);


        /*
            //gradcheck new batched CE:
            std::vector<int> labels = {0, 4, 2, 1};

            std::vector<Tensor*> leaves;
            leaves.emplace_back(mnist_model.W1); leaves.emplace_back(mnist_model.b1);
            leaves.emplace_back(mnist_model.W2); leaves.emplace_back(mnist_model.b2);

            tensorGraphCtx x_ctx;                       // outlives the gradcheck
            Tensor* x_master = x_ctx.make(4, 784);
            x_master->init_tensor_random(-1.0, 1.0);

            auto build = [labels, x_master](tensorGraphCtx& ctx, const std::vector<Tensor*>& leaves) {
                Tensor* x = clone(ctx, x_master);
                Tensor* logits = MNIST_MLP::mnist_forward_ops(ctx, x, leaves[0], leaves[1], leaves[2], leaves[3]);
                return Graph{cross_entropy_loss(ctx, logits, labels), leaves};
            };


            tensor_gradcheck(build, leaves,  30);


        */


        int steps = 60000;
        int batch_size = 64;
        int epochs = 5;

        std::cout << "Starting training for " << steps << " steps\n";

        for (int e = 0; e < epochs; e++) {
            auto start = std::chrono::steady_clock::now();

            for (int i = 0; i < steps / batch_size; ++i) {

                tensorGraphContext step_ctx;
                Tensor* ex_data = step_ctx.make(batch_size,784);

                std::vector<int> batch_labels(batch_size);
                for (int j = 0; j < batch_size; ++j) {
                    int idx = i * batch_size + j;
                    for (int p = 0; p < 784; ++p)
                        ex_data->data[j * 784 + p] = mnist.images[idx][p] / 255.0;
                    batch_labels[j] = static_cast<int>(mnist.labels[idx]);
                }

                Tensor* logits = mnist_model.forward(step_ctx, ex_data);
                Tensor* loss = cross_entropy_loss(step_ctx, logits, batch_labels);


                if (i % 100 == 0) {std::cout << "loss on batch " << i << ", " << "epoch " << e << " was " << loss->data[0] << std::endl;}
                backwards(loss);


                sgd_step(mnist_model.params, 0.05);
                zero_grad(mnist_model.params);
            }
            auto end = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Elapsed time for epoch " << e << " : " << elapsed_ms.count() << " ms\n";
        }




        //run inference:


        auto start_infer = std::chrono::steady_clock::now();
        int test_examples = 10000;
        auto mnist_test = load_mnist(DEEPC_DATA_DIR "/MNIST_data/t10k-images.idx3-ubyte",
                                DEEPC_DATA_DIR "/MNIST_data/t10k-labels.idx1-ubyte");

        std::cout << "Starting inference on " << test_examples << " examples\n";

        float accuracy_sum = 0.0;
        for (int i = 0; i < test_examples; ++i) {

            tensorGraphContext step_ctx;
            Tensor* ex_data = step_ctx.make(1,784);

            for (int p = 0; p < 784; ++p)
                ex_data->data[p] = mnist_test.images[i][p] / 255.0;

            if (mnist_model.infer(ex_data) == mnist_test.labels[i]) { ++accuracy_sum; };
        }

        auto end_infer = std::chrono::steady_clock::now();
        auto elapsed_ms_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer);

        std::cout << "Elapsed inference time: " << elapsed_ms_infer.count() << " ms\n";
        std::cout << "Accuracy: " << accuracy_sum / static_cast<float>(test_examples) << "\n";


        return 0;
    }
