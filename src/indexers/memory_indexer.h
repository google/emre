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

// convenience classes to easily construct a memory index

#ifndef EMRE_INDEXERS_MEMORY_INDEXER_H_  // NOLINT
#define EMRE_INDEXERS_MEMORY_INDEXER_H_

#include "memory_vector.h"  // NOLINT
#include "vector_indexer.h"  // NOLINT

namespace emre {

class MemoryIndexBuilder : public VectorIndexBuilder {
 public:
  explicit MemoryIndexBuilder(const std::string& feature_family,
                              int size_hint = 0)
      : VectorIndexBuilder(feature_family,
                           new MemoryVectorBuilderInt32(),
                           new MemoryVectorBuilder<std::string>(),
                           new MemoryVectorBuilder<int64>(),
                           new MemoryVectorBuilder<double>()) {}

  ~MemoryIndexBuilder() {}
};

class MemoryIndexReader : public VectorIndexReader {
 public:
  explicit MemoryIndexReader(VectorIndexReader* index) :
    VectorIndexReader(
        index->GetFeatureFamily(),
        new MemoryVectorReader<int32>(index->level_reader_.get())) {
    if (index->scaling_reader_ != nullptr) {
      scaling_reader_.reset(
          new MemoryVectorReader<double>(index->scaling_reader_.get()));
    }
    if (index->str_level_map_reader_ != nullptr) {
      str_level_map_reader_.reset(
          new MemoryVectorReader<std::string>(index->str_level_map_reader_.get()));
      str_level_map_itr_ = str_level_map_reader_->GetIterator();
    }
    if (index->int64_level_map_reader_ != nullptr) {
      int64_level_map_reader_.reset(new MemoryVectorReader<int64>(
          index->int64_level_map_reader_.get()));
      int64_level_map_itr_ = int64_level_map_reader_->GetIterator();
    }
  }
};

}  // namespace emre

#endif  // EMRE_INDEXERS_MEMORY_INDEXER_H_  // NOLINT
