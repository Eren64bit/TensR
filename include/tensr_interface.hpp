#pragma once

#include "tensr_metadata.hpp"

template <typename T>
class tensr_interface
{
protected:
    tensr_metadata metadata_;

public:
    tensr_interface(const std::vector<size_t> &shape, size_t offset = 0)
        : metadata_(shape, offset) {}

    tensr_interface(const tensr_metadata& meta) : metadata_(meta) {}
    // Data API
    virtual T *raw_data() = 0;
    virtual const T *raw_data() const = 0;

    virtual std::shared_ptr<T[]> data_owner() = 0;
    virtual std::shared_ptr<T[]> data_owner() const = 0;

    // Element API
    virtual T &at(const std::vector<size_t> &indices) = 0;
    virtual const T &at(const std::vector<size_t> &indices) const = 0;

    virtual T &operator()(const std::vector<size_t> &indices) = 0;
    virtual const T &operator()(const std::vector<size_t> &indices) const = 0;

    virtual T &operator[](size_t index) = 0;
    virtual const T &operator[](size_t index) const = 0;
};