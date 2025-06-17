#include "../include/tensr/Lens.hpp"
#include "../include/tensr/util/IndexUtils.hpp"

template<typename T>
TensrLens<T>::TensrLens(std::shared_ptr<std::vector<T>> data_ptr, std::vector<size_t> shape, std::vector<size_t> stride, size_t offset) 
    :   data_(data_ptr), shape_(std::move(shape)), stride_(std::move(stride)), offset_(offset) {

        total_size_ = compute_total_size(shape_);
        rank_ = compute_rank(shape_);

    }

template<typename T>
T& TensrLens<T>::at(const std::vector<size_t>& indices) {
    size_t flat_idx = flat_index(indices);
    auto data_ptr = data_.lock();
    if (!data_ptr) {
        throw std::runtime_error("Data pointer expired");
    }
    return (*data_ptr)[flat_idx + offset_];
}