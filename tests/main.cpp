
#include <iostream>
#include "../include/tensr/Tensr.hpp"
#include "../include/tensr/Lens.hpp"
#include "../include/tensr/LazyOps.hpp"
#include "../include/tensr/LazyEval.hpp"
#include "../include/tensr/util/TensrUtils.hpp"

int main() {
    // Tensor creation
    Tensr<float> a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensr<float> b({1, 3}, {10, 20, 30});

    std::cout << "Tensor a shape: ";
    for (auto s : a.shape()) std::cout << s << " ";
    std::cout << "\nTensor b shape: ";
    for (auto s : b.shape()) std::cout << s << " ";
    std::cout << "\n";

    // Broadcasting and lazy addition
    auto expr = a + b;
    auto result = evaluate<float>(expr);

    std::cout << "Result of a + b (broadcasted):\n";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << (*result.data().lock())[i] << " ";
    }
    std::cout << "\n";

    // Squeeze and unsqueeze
    squeeze(result);
    std::cout << "Shape after squeeze: ";
    for (auto s : result.shape()) std::cout << s << " ";
    std::cout << "\n";

    unSqueeze(result, 0);
    std::cout << "Shape after unsqueeze(0): ";
    for (auto s : result.shape()) std::cout << s << " ";
    std::cout << "\n";

    // Lens (view) example
    TensrLens<float> view(result.data().lock(), result.shape(), result.stride(), 0);
    std::cout << "First row via lens: ";
    for (size_t i = 0; i < view.shape()[1]; ++i) {
        std::cout << view.at({0, i}) << " ";
    }
    std::cout << "\n";

    return 0;
}