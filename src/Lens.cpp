#include "../include/tensr/Lens.hpp"

template<typename T>
TensrLens<T>::TensrLens(std::shared_ptr<std::vector<T>> data_ptr, std::vector<size_t> shape, std::vector<size_t> stride, size_t offset) 
    :   data_(data_ptr), shape_(std::move(shape)), stride_(std::move(stride)), offset_(offset) {
        
        total_size_ = compute_total_size(shape_);
        rank_ = compute_rank(shape_);

    }
