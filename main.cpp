#include <iostream>
#include <vector>
#include <memory>
#include "include/tensr.hpp"
#include "include/tensrLens.hpp"
#include "include/tensrOps_decl.hpp"
#include "include/tensrOps_imp.hpp"
#include "include/tensrBroadcast.hpp"
#include "include/tensrLazy.hpp"
#include "include/tensrSugar.hpp"

int main() {
    try {
        // 1. Tensor oluşturma ve doldurma
        std::vector<size_t> shape = {2, 3};
        tensr::Tensr<float> t(shape);
        t.fill(1.0f);
        for (size_t i = 0; i < t.size(); ++i) {
            auto idx = indexUtils::unflatten_index(i, shape);
            t.at(idx) = static_cast<float>(i + 1);
        }
        std::cout << "Tensr t info:\n";
        t.info();

        // 2. Lens oluşturma ve erişim
        auto locked_data = t.data().lock();
        tensrLens::lens<float> l(locked_data, t.shape(), t.stride(), t.offset());
        std::cout << "\nLens l info:\n";
        l.info();
        std::cout << "Lens l(1,2): " << l(1,2) << std::endl;
        std::cout << "Lens l.at({0,1}): " << l.at({0,1}) << std::endl;

        // 3. Transpose
        auto lT = l.transpose({1,0});
        std::cout << "\nTransposed lens info:\n";
        lT.info();

        // 4. Slice
        std::vector<tensrOps::SliceRange> ranges = { {0,2,1}, {1,3,1} };
        auto lS = l.slice(ranges);
        std::cout << "\nSliced lens info:\n";
        lS.info();

        // 5. Flatten ve reshape
        auto lF = l.flatten();
        std::cout << "\nFlattened lens info:\n";
        lF.info();
        auto lR = lF.reshape({2,3});
        std::cout << "\nReshaped lens info:\n";
        lR.info();

        // 6. Copy ve fill
        auto t2 = l.copy();
        std::cout << "\nCopied tensr info:\n";
        t2.info();
        t2.fill(42.0f);
        std::cout << "After fill, t2.at({0,0}): " << t2.at({0,0}) << std::endl;

        // 7. Broadcast
        std::vector<size_t> bshape = {2, 3, 4};
        tensr::Tensr<float> tb({1,3,1});
        for (size_t i = 0; i < tb.size(); ++i)
            tb.at(indexUtils::unflatten_index(i, {1,3,1})) = float(i+1);
        auto blens = broadcast::broadcast_to(tb, bshape);
        std::cout << "\nBroadcasted lens info:\n";
        blens.info();
        std::cout << "Broadcasted lens (1,2,3): " << blens(1,2,3) << std::endl;

        // 8. Squeeze ve Unsqueeze
        tensr::Tensr<float> t_sq({1,2,3});
        auto sq = t_sq.squeeze();
        std::cout << "\nSqueezed tensor info:\n";
        sq.info();
        auto usq = sq.unsqueeze(0);
        std::cout << "\nUnsqueezed tensor info:\n";
        usq.info();

        // 9. Cache test
        l.cache_data_ptr();
        std::cout << "\nAfter caching, lens info:\n";
        l.info();
        l.clear_cache();

        // 10. Lazy evaluation (tensrLazy)
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

        // 11. Sugar operatorleri (tensrSugar)
        auto sum = t + t3;
        std::cout << "\nSugar operator '+' result info:\n";
        sum.info();
        auto diff = t - t3;
        std::cout << "\nSugar operator '-' result info:\n";
        diff.info();
        auto mul = t * t3;
        std::cout << "\nSugar operator '*' result info:\n";
        mul.info();
        auto div = t3 / t;
        std::cout << "\nSugar operator '/' result info:\n";
        div.info();

        // 12. Hatalı erişim ve reshape testleri
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