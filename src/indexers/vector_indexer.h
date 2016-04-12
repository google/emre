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

#ifndef EMRE_INDEXERS_VECTOR_INDEXER_H_  // NOLINT
#define EMRE_INDEXERS_VECTOR_INDEXER_H_

#include <algorithm>
#include <memory>
#include <string>

#include "indexer.h"  // NOLINT
#include "vector_storage.h"  // NOLINT

namespace emre {

class VectorIndexReader : public IndexReaderWithDefaultScaling {
 private:
  explicit VectorIndexReader(const string& feature_family, int num_rows)
      : IndexReaderWithDefaultScaling(num_rows),
        feature_family_(feature_family) {}

 public:
  VectorIndexReader(const string& feature_family,
                    VectorReader<int32>* level_reader);

  virtual ~VectorIndexReader() {}

  int GetNumLevels() const {
    if (str_level_map_reader_ == nullptr &&
        int64_level_map_reader_ == nullptr) {
      return 0;
    } else {
      return str_level_map_reader_ != nullptr ? str_level_map_reader_->Size()
                                              : int64_level_map_reader_->Size();
    }
  }

  const string GetFeatureFamily() const override { return feature_family_; }

  VectorReader<int32>::Iterator GetLevelIterator() override {
    return level_reader_->GetIterator();
  }

  VectorReader<double>::Iterator GetScalingIterator() override {
    if (scaling_reader_.get() != nullptr) {
      return scaling_reader_->GetIterator();
    } else {
      return IndexReaderWithDefaultScaling::GetScalingIterator();
    }
  }

  int GetNumObservations() const override {
    assert(level_reader_.get() != nullptr);
    return level_reader_->Size();
  }

  bool HasNonTrivialScaling() const {
    return scaling_reader_.get() != nullptr
           && scaling_reader_->Size() == GetNumObservations();
  }

  // FillFeatureLevel is based on the VectorReader interface,
  // it reads through its elements sequentially, hence we need to
  // reset it with 'ResetFeatureLevel' before actual use.
  void FillFeatureLevel(FeatureData* data) override;
  void ResetFeatureLevel() override {
    if (str_level_map_reader_ != nullptr) {
      str_level_map_itr_.Reset();
    }
    if (int64_level_map_reader_ != nullptr) {
      int64_level_map_itr_.Reset();
    }
  }

 private:
  friend class MemoryIndexReader;  // MemoryIndexReader constructor needs access
  friend class VectorIndexBuilder;  // MoveToReader needs access to privates

  const string feature_family_;
  // Level reader
  std::unique_ptr<VectorReader<int32>> level_reader_;
  // Scaling reader
  std::unique_ptr<VectorReader<double>> scaling_reader_;
  // Fields to map ID to level. Vector index is ID.
  // Note: Only one field is used for a specific feature family.
  std::unique_ptr<VectorReader<string>> str_level_map_reader_;
  std::unique_ptr<VectorReader<int64>> int64_level_map_reader_;
  // Iterators for above two VectorReaders.
  VectorReader<string>::Iterator str_level_map_itr_;
  VectorReader<int64>::Iterator int64_level_map_itr_;
};

class VectorIndexBuilder : public IndexBuilder {
 public:
  VectorIndexBuilder(const string& feature_family,
                     VectorBuilder<int32>* level_writer,
                     VectorBuilder<string>* str_level_map_writer,
                     VectorBuilder<int64>* int64_level_map_writer,
                     VectorBuilder<double>* scaling_writer);
  virtual ~VectorIndexBuilder() {}

  const string GetFeatureFamily() const override { return feature_family_; }

  void ProcessData(const FeatureData& data) override;

  std::unique_ptr<IndexReader> MoveToReader() override;

  int GetFeatureLevelId(const FeatureData& data) const override;

  void WriteLevelId(int level_id) {
    level_writer_->Write(level_id);
  }

  void MayWriteScaling(double scaling) {
    if (scaling_writer_.get() != nullptr) scaling_writer_->Write(scaling);
  }

  int GetStrLevelId(const string& level) {
    auto ret = str_level_to_id_.insert(std::make_pair(level, NextLevelId()));
    if (ret.second) {
      str_level_map_writer_->Write(level);
    }
    return ret.first->second;
  }

  int GetInt64LevelId(int64 level) {
    auto ret = int64_level_to_id_.insert(std::make_pair(level, NextLevelId()));
    if (ret.second) {
      int64_level_map_writer_->Write(level);
    }
    return ret.first->second;
  }

  int NextLevelId() {
    return std::max(str_level_map_writer_->Size(),
                    int64_level_map_writer_->Size());
  }

 private:
  const string feature_family_;

  // Level writer
  std::unique_ptr<VectorBuilder<int32>> level_writer_;
  // Scaling writer
  std::unique_ptr<VectorBuilder<double>> scaling_writer_;
  // Fields used to allocate a ID to each level of the feature family.
  // Note: A specific feature family contains either string levels or int64
  // levels, so either str_level_to_id_ or int64_level_to_id_ is
  // used for a specific feature family.
  std::unordered_map<string, int> str_level_to_id_;
  std::unique_ptr<VectorBuilder<string>> str_level_map_writer_;
  std::unordered_map<int64, int> int64_level_to_id_;
  std::unique_ptr<VectorBuilder<int64>> int64_level_map_writer_;
};

}  // namespace emre

#endif  // EMRE_INDEXERS_VECTOR_INDEXER_H_  // NOLINT
