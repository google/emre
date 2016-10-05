// Copyright 2012-2016 Google
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _EMRE_UTIL_ARRAYSLICE_H_  // NOLINT
#define _EMRE_UTIL_ARRAYSLICE_H_

#include <vector>

namespace emre {
namespace util {

template <typename T>
class MutableArraySlice {
 public:
  typedef T val_type;

  MutableArraySlice() : ptr_(nullptr), length_(0) {}
  MutableArraySlice(const MutableArraySlice& x, int pos, int len)
      : ptr_(x.ptr_ + pos), length_(len) {}
  MutableArraySlice(T* ptr, int length) : ptr_(ptr), length_(length) {}
  MutableArraySlice(std::vector<val_type>* vec)
      : ptr_(vec->data()), length_(vec->size()) {}

  int size() const { return length_; }
  T* data() const { return ptr_; }

  T& operator[](int i) const { return ptr_[i]; }
  T* begin() const { return ptr_; }
  T* end() const { return ptr_ + length_; }

 private:
  T* ptr_;
  int length_;
};

template <typename T>
class ArraySlice {
 public:
  typedef T val_type;

  ArraySlice() : ptr_(nullptr), length_(0) {}
  ArraySlice(const ArraySlice& x, int pos, int len)
      : ptr_(x.ptr_ + pos), length_(len) {}
  ArraySlice(const T* ptr, int length) : ptr_(ptr), length_(length) {}

  ArraySlice(const std::vector<val_type>& vec)
      : ptr_(vec.data()), length_(vec.size()) {}

  ArraySlice(const MutableArraySlice<val_type>& V)
      : ptr_(V.data()), length_(V.size()) {}

  int size() const { return length_; }
  const T* data() const { return ptr_; }

  const T& operator[](int i) const { return ptr_[i]; }
  const T* begin() const { return ptr_; }
  const T* end() const { return ptr_ + length_; }

  bool operator==(ArraySlice<T> other) const {
    if (size() != other.size()) return false;
    if (data() == other.data()) return true;
    return std::equal(data(), data() + size(), other.data());
  }

 private:
  const T* ptr_;
  int length_;
};

}  // namespace util
}  // namespace emre

#endif  // _EMRE_UTIL_ARRAYSLICE_H_  // NOLINT
