#include <iostream>
#include "include/tensr.hpp"

int main() {

    std::vector<size_t> shape = {2, 3};
    tensr::Tensr<int> t1(shape);

    std::cout << "t1 shape: ";
    for (auto s : t1.shape()) std::cout << s << " ";
    std::cout << "\nt1 size: " << t1.size() << std::endl;


    std::vector<size_t> shape2 = {2, 2};
    std::vector<int> data = {1, 2, 3, 4};
    tensr::Tensr<int> t2(shape2, data);

    std::cout << "t2 shape: ";
    for (auto s : t2.shape()) std::cout << s << " ";
    std::cout << "\nt2 size: " << t2.size() << std::endl;

    

    return 0;
}