#pragma once

#include "tensr.hpp"

namespace tensrLens{

template<typename T>
class lens : public tensr::TensrBase<T> {
private:
    std::weak_ptr<std::vector<size_t>> data_ptr_;

    std::vector<size_t> shape_;
    std::vector<size_t> stride_;

    size_t offset_;
    size_t total_size_;
    size_t rank_;
};

}