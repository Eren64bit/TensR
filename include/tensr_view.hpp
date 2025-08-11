#pragma once

#include "tensr_static.hpp"

template <typename T>
class tensr_view
{
  tensr_metadata view_meta_;
  std::shared_ptr<T[]> data_owner_;
  T *data_ptr_;

public:
  tensr_view(const tensr_metadata &view_meta,
             std::shared_ptr<T[]> data_ptr_owner)
      : view_meta_(view_meta), data_owner_(data_ptr_owner),
        data_ptr_(data_ptr_owner.get() + view_meta.offset()) {}
};
