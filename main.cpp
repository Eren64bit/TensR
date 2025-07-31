#include <iostream>
#include "include/tensrREAL.h"
#include "include/tensrVIEW.h"

int main() {

    tensrREAL<float> tensor(3, 4, 5);

    std::cout << "Tensor shape: ";
    for (auto s : tensor.shape()) std::cout << s << " ";
    std::cout << "\nTensor size: " << tensor.size() << std::endl;


    std::cout << "Tensor backend: " << (tensor.backend_ == backendType::GPU ? "GPU" : "CPU") << std::endl;


    auto data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i) (*data)[i] = static_cast<float>(i);

    std::cout << "Tensor first element: " << tensor.at({0,0,0}) << std::endl;
    std::cout << "Tensor last element: " << tensor.at({2,3,4}) << std::endl;


    tensrVIEW<float> view(data, 3, 4, 5);
    std::cout << "View shape: ";
    for (auto s : view.shape()) std::cout << s << " ";
    std::cout << "\nView first element: " << view.at({0,0,0}) << std::endl;


    tensor.set_backend(backendType::CPU);
    std::cout << "Tensor backend (after set_backend): " << (tensor.backend_ == backendType::GPU ? "GPU" : "CPU") << std::endl;


    try {
        tensor.set_backend(backendType::GPU);
        tensor.allocate_gpu();
        std::cout << "GPU memory allocated." << std::endl;
        tensor.free_gpu();
        std::cout << "GPU memory freed." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "GPU operation error: " << e.what() << std::endl;
    }

    return 0;
}