#include "../headers/mnist_loader.h"
#include <fstream>
#include <stdexcept>

static uint32_t read_be_uint32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) |
           (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8)  |
            uint32_t(b[3]);
}

MNIST load_mnist(const std::string& image_path, const std::string& label_path) {
    std::ifstream img(image_path, std::ios::binary);
    std::ifstream lbl(label_path, std::ios::binary);
    if (!img || !lbl) throw std::runtime_error("Could not open MNIST files");

    MNIST data;

    if (read_be_uint32(img) != 2051) throw std::runtime_error("Bad image magic");
    uint32_t num_images = read_be_uint32(img);
    data.rows = static_cast<int>(read_be_uint32(img));
    data.cols = static_cast<int>(read_be_uint32(img));

    if (read_be_uint32(lbl) != 2049) throw std::runtime_error("Bad label magic");
    uint32_t num_labels = read_be_uint32(lbl);

    if (num_images != num_labels) throw std::runtime_error("Image/label count mismatch");

    data.images.resize(num_images, std::vector<uint8_t>(data.rows * data.cols));
    data.labels.resize(num_labels);

    for (uint32_t i = 0; i < num_images; ++i) {
        img.read(reinterpret_cast<char*>(data.images[i].data()), data.rows * data.cols);
        lbl.read(reinterpret_cast<char*>(&data.labels[i]), 1);
    }

    return data;
}
