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

#include "memory_vector.h"  // NOLINT

#include <algorithm>
#include <functional>
#include <memory>

namespace emre {

static const int kMaxBool = 1;

MemoryVectorBuilderInt32::MemoryVectorBuilderInt32() :  max_level_(1) {}

void MemoryVectorBuilderInt32::UpgradeVector(int old_max, int new_max) {
  if (old_max <= kMaxBool && new_max > kMaxBool) {
    vec_uint8_.resize(vec_bool_.size());
    std::copy(vec_bool_.begin(), vec_bool_.end(), vec_uint8_.begin());
    // destroy vec_bool_
    vector<bool> dummy;
    vec_bool_.swap(dummy);
  }
  if (old_max <= kuint8max && new_max > kuint8max) {
    vec_uint16_.resize(vec_uint8_.size());
    std::copy(vec_uint8_.begin(), vec_uint8_.end(), vec_uint16_.begin());
    // destroy vec_uint8_
    vector<uint8> dummy;
    vec_uint8_.swap(dummy);
  }
  if (old_max <= kuint16max && new_max > kuint16max) {
    vec_int32_.resize(vec_uint16_.size());
    std::transform(vec_uint16_.begin(), vec_uint16_.end(), vec_int32_.begin(),
                   [](uint16 v) -> int32 {
                     return static_cast<int32>(v) - 1; });
    // destroy vec_uint16_
    vector<uint16> dummy;
    vec_uint16_.swap(dummy);
  }
}

void MemoryVectorBuilderInt32::Write(int32 x) {
  int old_max_level = max_level_;
  max_level_ = std::max(max_level_, x + 1);

  // we potentially have to "upgrade" the vector
  if (x + 1 > old_max_level && (
      (old_max_level <= kMaxBool && max_level_ > kMaxBool) ||
      (old_max_level <= kuint8max && max_level_ > kuint8max) ||
      (old_max_level <= kuint16max && max_level_ > kuint16max))) {
    this->UpgradeVector(old_max_level, max_level_);
  }

  // add element to the proper vector
  if (max_level_ > kuint16max) {
    vec_int32_.push_back(x);
  } else if (max_level_ > kuint8max) {
    vec_uint16_.push_back(static_cast<uint16>(x + 1));
  } else if (max_level_ > kMaxBool) {
    vec_uint8_.push_back(static_cast<uint8>(x + 1));
  } else {
    vec_bool_.push_back(static_cast<bool>(x + 1));
  }
}

std::unique_ptr<VectorReader<int32>> MemoryVectorBuilderInt32::MoveToReader() {
  if (!vec_int32_.empty()) {
    auto reader = new MemoryVectorReader<int32>();
    vec_int32_.swap(reader->vec_);
    return std::unique_ptr<VectorReader<int32>>(reader);
  } else if (!vec_uint16_.empty()) {
    auto reader = new CastedMemoryVectorReader<uint16>();
    vec_uint16_.swap(reader->vec_);
    return std::unique_ptr<VectorReader<int32>>(reader);
  } else if (!vec_uint8_.empty()) {
    auto reader = new CastedMemoryVectorReader<uint8>();
    vec_uint8_.swap(reader->vec_);
    return std::unique_ptr<VectorReader<int32>>(reader);
  } else if (!vec_bool_.empty()) {
    auto reader = new CastedMemoryVectorReader<bool>();
    vec_bool_.swap(reader->vec_);
    return std::unique_ptr<VectorReader<int32>>(reader);
  }

  // LOG(ERROR) << "Vector builder has no features";
  return std::unique_ptr<VectorReader<int32>>(nullptr);
}

}  // namespace emre
