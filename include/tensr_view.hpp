#pragma once

#include "tensr_static.hpp"

enum class view_type {
  SLICE,   // Represents a slice of the tensor
  STRIDED, // Represents a strided view of the tensor
  MASKED,  // Represents a masked view of the tensor
  INDEXED, // Represents an indexed view of the tensor
  CUSTOM   // Represents a custom view of the tensor
};

class tensr_view_base {
protected:
  view_type type_;
  tensr_metadata parent_metadata_;

public:
  virtual tensr_metadata get_view_metadata() const = 0;
  virtual bool is_contiguous() const = 0;
};

template <typename T, typename allocator = default_smart_allocator<T>>
class tensr_view {
  std::unique_ptr<tensr_view_base> view_info_;

public:
};
