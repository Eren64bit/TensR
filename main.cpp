#include <iostream>
#include "include/tensr_static.hpp"

int main() {

    std::vector<size_t> tnsr_shape = {3, 2};
    tensr_static<float> my_tensor(tnsr_shape);
    my_tensor.fill(1.2);
    my_tensor.info();
    my_tensor.visualize();

    auto reshaped_tnsr = my_tensor.reshape({6,1});
    reshaped_tnsr.info();

}