#include <iostream>
#include "include/tensr_static.h"

int main() {

    std::vector<size_t> shape = {2, 3};
    tensr_static<float> tensor(shape);


    tensor.fill_zeros();


    std::cout << "Tensor elements after fill_zeros:\n";
    for (size_t i = 0; i < tensor.size(); ++i) {
        std::cout << tensor[i] << " ";
    }
    std::cout << std::endl;


    tensor.fill_custom(7.5f);

    std::cout << "Tensor elements after fill_custom(7.5):\n";
    for (size_t i = 0; i < tensor.size(); ++i) {
        std::cout << tensor[i] << " ";
    }
    std::cout << std::endl;


    std::vector<size_t> idx = {1, 2};
    tensor.at(idx) = 42.0f;
    std::cout << "tensor.at({1,2}) = " << tensor.at(idx) << std::endl;
    std::cout << "tensor(1, 2) = " << tensor(idx) << std::endl;
    std::cout << "tensor[5] = " << tensor[5] << std::endl;

    return 0;
}