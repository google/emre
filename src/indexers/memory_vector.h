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

#ifndef EMRE_INDEXERS_MEMORY_VECTOR_H_  // NOLINT
#define EMRE_INDEXERS_MEMORY_VECTOR_H_

#include <algorithm>

#include "vector_storage.h"  // NOLINT
#include "base/integral_types.h"

namespace emre {

//  thin wrappers around STL vectors
template<typename T>
class MemoryVectorBuilder;

template<typename T>
class MemoryVectorReader : public VectorReader<T> {
 public:
  MemoryVectorReader() { iter_ = vec_.begin(); }
  // very convenient for unit testing
  explicit MemoryVectorReader(const vector<T>& init_vec) {
    vec_.resize(init_vec.size());
    std::copy(init_vec.begin(), init_vec.end(), vec_.begin());
    iter_ = vec_.begin();
  }
  // useful for loading pre-processed data into memory
  explicit MemoryVectorReader(VectorReader<T>* vector_reader) {
    int size = vector_reader->Size();
    vec_.resize(size);
    auto iter = vector_reader->GetIterator();
    for (int i = 0; i < size && !iter.Done(); ++i) {
      vec_[i] = iter.Next();
    }
    iter_ = vec_.begin();
  }

  int Size() const override { return vec_.size(); }

 protected:
  const vector<T>* AsVector() const override { return &vec_; }
  void Reset() override { iter_ = vec_.begin(); }
  T Next() override { return *(iter_++); }

 private:
  friend class MemoryVectorBuilder<T>;
  friend class MemoryVectorBuilderInt32;

  typename vector<T>::const_iterator iter_;
  vector<T> vec_;
};

template<typename T>
class MemoryVectorBuilder : public VectorBuilder<T> {
 public:
  MemoryVectorBuilder() {}

  void Write(T x) override { return vec_.push_back(x); }
  void Finish() override {}
  std::unique_ptr<VectorReader<T>> MoveToReader() {
    auto* reader = new MemoryVectorReader<T>();
    vec_.swap(reader->vec_);
    std::unique_ptr<VectorReader<T>> reader_ptr(reader);
    return reader_ptr;
  }
  int Size() const override { return vec_.size(); }

 private:
  vector<T> vec_;
};

template <typename T>
class CastedMemoryVectorReader : public VectorReader<int32> {
 public:
  CastedMemoryVectorReader() { iter_ = vec_.begin(); }
  int Size() const override { return vec_.size(); }

 protected:
  const vector<int32>* AsVector() const override { return nullptr; }
  void Reset() override { iter_ = vec_.begin(); }
  int32 Next() override { return static_cast<int32>(*(iter_++)) - 1; }

 private:
  friend class MemoryVectorBuilderInt32;

  typename vector<T>::const_iterator iter_;
  vector<T> vec_;
};

class MemoryVectorBuilderInt32 : public VectorBuilder<int32> {
 public:
  MemoryVectorBuilderInt32();
  virtual ~MemoryVectorBuilderInt32() {}

  void Write(int32 x) override;
  void Finish() override {}
  std::unique_ptr<VectorReader<int32>> MoveToReader() override;
  int Size() const override {
    return std::max(
        std::max(vec_int32_.size(), vec_uint16_.size()),
        std::max(vec_uint8_.size(), vec_bool_.size()));
  }

 private:
  void UpgradeVector(int old_max_level, int new_max_level);

 private:
  vector<int32> vec_int32_;
  vector<uint16> vec_uint16_;
  vector<uint8> vec_uint8_;
  vector<bool> vec_bool_;
  int max_level_;
};

}  // namespace emre

#endif  // EMRE_INDEXERS_MEMORY_VECTOR_H_  // NOLINT
