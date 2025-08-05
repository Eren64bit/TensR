

#include "../include/tensr_static.h"
#include "../include/tensr_view.h"


template<typename T>
tensr_view<T> tensr_static<T>::slice(const std::vector<size_t>& start, const std::vector<size_t>& end)  const{
    if (start.size() != shape_.size() || end.size() != shape_.size()) {
        throw std::invalid_argument("Start and end indices must match the tensor's rank.");
    }
    
    std::vector<size_t> new_shape(shape_);
    std::vector<size_t> new_strides(strides_);
    
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (start[i] >= end[i] || end[i] > shape_[i]) {
            throw std::out_of_range("Invalid slice range.");
        }
        new_shape[i] = end[i] - start[i];
    }
    
    size_t new_offset = tensr_utils::flatten_index(shape_, strides_, start);
    
    return tensr_view<T>(data_, new_shape, new_offset);
}