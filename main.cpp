#include <iostream>
#include "include/tensr.hpp"
#include "include/tensrLens.hpp"

int main() {
    // İlk tensor testi
    std::vector<size_t> shape = {2, 3};
    tensr::Tensr<int> t1(shape);
    
    std::cout << "t1 shape: ";
    for (auto s : t1.shape()) std::cout << s << " ";
    std::cout << "\nt1 size: " << t1.size() << std::endl;
    
    std::cout << "t1 stride: ";
    for (auto s : t1.stride()) std::cout << s << " ";
    std::cout << std::endl;
    
    // İkinci tensor testi
    std::vector<size_t> shape2 = {2, 2};
    std::vector<int> data = {1, 2, 3, 4};
    tensr::Tensr<int> t2(shape2, data);
    
    std::cout << "t2 shape: ";
    for (auto s : t2.shape()) std::cout << s << " ";
    std::cout << "\nt2 size: " << t2.size() << std::endl;
    
    std::cout << "t2 stride: ";
    for (auto s : t2.stride()) std::cout << s << " ";
    std::cout << std::endl;
    
    // Eleman atama ve erişim testi
    std::cout << "t1.at({0,0}) Before: " << t1.at({0, 0}) << std::endl;
    t1.at({0, 0}) = 42;
    std::cout << "t1.at({0,0}) after: " << t1.at({0, 0}) << std::endl;
    
    // t2 için test
    std::cout << "t2.at({0,0}): " << t2.at({0, 0}) << std::endl;
    std::cout << "t2.at({0,1}): " << t2.at({0, 1}) << std::endl;
    std::cout << "t2.at({1,0}): " << t2.at({1, 0}) << std::endl;
    std::cout << "t2.at({1,1}): " << t2.at({1, 1}) << std::endl;

    std::vector<size_t> shape123 = {2, 3};
    tensr::Tensr<int> t9(shape123);

    // Veriye değer ata
    t9.at({0, 0}) = 10;
    t9.at({0, 1}) = 20;
    t9.at({1, 2}) = 99;

    // Tensörün datasını paylaşan bir lens oluştur
    auto lens1 = tensrLens::lens<int>(
        t9.data().lock(), // shared_ptr
        t9.shape(),
        t9.stride(),
        t9.offset()
    );

    // Lens ile veri okuma/yazma
    std::cout << "Lens ile erişim: lens1.at({0,0}) = " << lens1.at({0, 0}) << std::endl;
    lens1.at({1, 2}) = 123;
    std::cout << "Lens ile yazdiktan sonra t9.at({1,2}) = " << t9.at({1, 2}) << std::endl;

    // Lens info fonksiyonunu test et
    lens1.info();

    // Cache test
    lens1.cache_data_ptr();
    std::cout << "Cache sonrasi lens1.at({0,1}) = " << lens1.at({0, 1}) << std::endl;
    lens1.clear_cache();

    
    return 0;
}