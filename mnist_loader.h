#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct MNIST {
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;
    int rows = 0, cols = 0;
};

MNIST load_mnist(const std::string& image_path, const std::string& label_path);