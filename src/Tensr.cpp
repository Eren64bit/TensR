#include "../include/tensr/Tensr.hpp"
#include "../include/tensr/util/TensrUtils.hpp"

//------------------Constructers
template<typename T>
Tensr<T>::Tensr(std::vector<size_t> shape) 
    : shape_(std::move(shape))  {
    this->total_size_ = compute_total_size(shape);
    this->stride_ = compute_strides(shape);
    this->rank_ = compute_rank(shape);

    data_.resize(total_size_);
    std::fill(data_.begin(), data_.end(), 0);
}

template<typename T>
Tensr<T>::Tensr(std::vector<size_t> shape, std::vector<value_type> data) 
    : shape_(std::move(shape)), data_(std::move(data))  {
    this->total_size_ = compute_total_size(shape);

    if (total_size_ != data.size()) {
        throw std::runtime_error("Data and Shape does not match up");
    }
    
    this->stride_ = compute_strides(shape);
    this->rank_ = compute_rank(shape);


}
//------------------Data()
template<typename T>
const std::weak_ptr<T> Tensr<T>::data() {

}