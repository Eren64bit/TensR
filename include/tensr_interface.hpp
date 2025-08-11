#pragma once

#include "tensr_metadata.hpp"

template <typename T>
class tensr_interface
{
protected:
    tensr_metadata metadata_;

public:
    // Data API
    virtual T *raw_data() = 0;
    virtual const T *raw_data() const = 0;

    virtual std::shared_ptr<T[]> data_owner() = 0;
    virtual const std::shared_ptr<T[]> data_owner() const = 0;

    // Element API
    virtual T &at(const std::vector<size_t> indices) = 0;
    virtual const T &at(const std::vector<size_t> indices) const = 0;

    virtual T &operator()(const std::vector<size_t> indices) = 0;
    virtual const T &operator()(const std::vector<size_t> indices) const = 0;

    virtual T &operator[](size_t index) = 0;
    virtual const T &operator[](size_t index) const = 0;

    // Metadata API
    const tensr_metadata &metadata() const { return metadata_; }
    const std::vector<size_t> &shape() const { return metadata_.shape(); }
    size_t size() const { return metadata_.size(); }
    size_t rank() const { return metadata_.rank(); }
};