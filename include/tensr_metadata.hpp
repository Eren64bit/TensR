#pragma once
#include "tensr_utils.hpp"

class tensr_metadata {

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_;
    size_t total_size_;
    bool is_contiguous_ = true;
    
public:

    tensr_metadata(const std::vector<size_t>& shape, size_t offset = 0)
        : shape_(shape), offset_(offset) {
            if (shape.empty()) throw std::invalid_argument("Shape must not be empty.");
            strides_ = tensr_utils::compute_strides(shape_);
            total_size_ = size();
        }

    [[nodiscard]] const std::vector<size_t>& shape() const { return shape_; }
    [[nodiscard]] const std::vector<size_t>& strides() const { return strides_; }

    [[nodiscard]] size_t size() const { return tensr_utils::compute_size(shape_); }
    [[nodiscard]] size_t rank() const { return shape_.size(); }

    [[nodiscard]] size_t offset() const { return offset_; }
    [[nodiscard]] size_t total_size() const { return total_size_; }

    void set_offset(size_t offset) { offset_ = offset; }

    size_t flatten_index(const std::vector<size_t>& indices) const {
        return tensr_utils::flatten_index(shape_, strides_, indices) + offset_;
    }
};