#include <iostream>
#include <vector>
#include <memory>
#include "include/tensr.hpp"
#include "include/tensrLens.hpp"
#include "include/tensrOps_decl.hpp"
#include "include/tensrOps_imp.hpp"
#include "include/tensrBroadcast.hpp"
#include "include/tensrLazy.hpp"

int main() {
    try {
        // 1. Tensor oluşturma ve doldurma
        std::vector<size_t> shape = {2, 3};
        tensr::Tensr<float> t(shape);
        for (size_t i = 0; i < t.size(); ++i) {
            auto idx = indexUtils::unflatten_index(i, shape);
            t.at(idx) = static_cast<float>(i + 1);
        }
        std::cout << "Tensr t info:\n";
        t.info();

        // 2. Lens oluşturma
        auto locked_data = t.data().lock();
        if (!locked_data) throw std::runtime_error("Tensor data expired");
        tensrLens::lens<float> l(locked_data, t.shape(), t.stride(), t.offset());
        std::cout << "\nLens l info:\n";
        l.info();

        // 3. Lens ile erişim
        std::cout << "\nLens l(1,2): " << l(1,2) << std::endl;
        std::cout << "Lens l.at({0,1}): " << l.at({0,1}) << std::endl;

        // 4. Transpose
        auto lT = l.transpose({1,0});
        std::cout << "\nTransposed lens info:\n";
        lT.info();
        std::cout << "lT(2,1): " << lT(2,1) << std::endl;

        // 5. Slice
        std::vector<tensrOps::SliceRange> ranges = { {0,2,1}, {1,3,1} };
        auto lS = l.slice(ranges);
        std::cout << "\nSliced lens info:\n";
        lS.info();
        std::cout << "lS(1,1): " << lS(1,1) << std::endl;

        // 6. Flatten ve reshape
        auto lF = l.flatten();
        std::cout << "\nFlattened lens info:\n";
        lF.info();
        auto lR = lF.reshape({2,3});
        std::cout << "\nReshaped lens info:\n";
        lR.info();

        // 7. Copy
        auto t2 = l.copy();
        std::cout << "\nCopied tensr info:\n";
        t2.info();

        // 8. Broadcast
        std::vector<size_t> bshape = {2, 3, 4};
        tensr::Tensr<float> tb({1,3,1});
        for (size_t i = 0; i < tb.size(); ++i)
            tb.at(indexUtils::unflatten_index(i, {1,3,1})) = float(i+1);
        auto blens = broadcast::broadcast_to(tb, bshape);
        std::cout << "\nBroadcasted lens info:\n";
        blens.info();
        std::cout << "Broadcasted lens (1,2,3): " << blens(1,2,3) << std::endl;

        // 9. Squeeze ve Unsqueeze
        auto sq = tensrOps::squeeze(t);
        std::cout << "\nSqueezed tensor info:\n";
        sq.info();
        auto usq = tensrOps::unsqueeze(t, 0);
        std::cout << "\nUnsqueezed tensor info:\n";
        usq.info();

        // 10. Cache test
        l.cache_data_ptr();
        std::cout << "\nAfter caching, lens info:\n";
        l.info();
        l.clear_cache();

        // 11. Lazy evaluation (tensrLazy)
        tensr::Tensr<float> t3(shape);
        for (size_t i = 0; i < t3.size(); ++i) {
            auto idx = indexUtils::unflatten_index(i, shape);
            t3.at(idx) = static_cast<float>(10 * (i + 1));
        }
        auto expr = tensrLazy::leaf(t) + tensrLazy::leaf(t3);
        auto result = tensrLazy::materialize(*expr);
        std::cout << "\nLazy add materialized result info:\n";
        result.info();
        std::cout << "Lazy add result (1,2): " << result.at({1,2}) << std::endl;

        // 12. Hatalı erişim testleri
        try {
            l(5,0); // Hatalı indeks
        } catch (const std::exception& e) {
            std::cout << "\nExpected error (out of bounds): " << e.what() << std::endl;
        }

        try {
            l.reshape({4,2}); // Hatalı reshape
        } catch (const std::exception& e) {
            std::cout << "Expected error (reshape): " << e.what() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Unexpected error: " << e.what() << std::endl;
    }
    return 0;
}