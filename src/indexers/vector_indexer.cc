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

#include "vector_indexer.h"  // NOLINT

namespace emre {

VectorIndexReader::VectorIndexReader(const std::string& feature_family,
                                     VectorReader<int32_t>* level_reader)
    : IndexReaderWithDefaultScaling(level_reader->Size()),
      feature_family_(feature_family),
      level_reader_(level_reader) {}

void VectorIndexReader::FillFeatureLevel(FeatureData* data) {
  if (str_level_map_reader_ != nullptr) {
    data->SetFeature(feature_family_, str_level_map_itr_.Next());
  } else {
    data->SetFeatureInt64(feature_family_, int64_level_map_itr_.Next());
  }
}

VectorIndexBuilder::VectorIndexBuilder(const std::string& feature_family,
                                       VectorBuilder<int32>* level_writer,
                                       VectorBuilder<std::string>* str_map_writer,
                                       VectorBuilder<int64>* int64_map_writer,
                                       VectorBuilder<double>* scaling_writer)
    : feature_family_(feature_family),
      level_writer_(level_writer),
      scaling_writer_(scaling_writer),
      str_level_map_writer_(str_map_writer),
      int64_level_map_writer_(int64_map_writer) {}

std::unique_ptr<IndexReader> VectorIndexBuilder::MoveToReader() {
  auto* reader = new VectorIndexReader(feature_family_, level_writer_->Size());

  assert(level_writer_.get() != nullptr);
  level_writer_->Finish();
  reader->level_reader_.reset(level_writer_->MoveToReader().release());

  assert(scaling_writer_.get() != nullptr);
  scaling_writer_->Finish();
  if (scaling_writer_->Size() > 0) {
    reader->scaling_reader_.reset(scaling_writer_->MoveToReader().release());
  }

  assert(str_level_map_writer_.get() != nullptr);
  str_level_map_writer_->Finish();
  if (str_level_map_writer_->Size() > 0) {
    reader->str_level_map_reader_.reset(str_level_map_writer_->
                                        MoveToReader().release());
    reader->str_level_map_itr_ = reader->str_level_map_reader_->GetIterator();
  }

  assert(int64_level_map_writer_.get());
  int64_level_map_writer_->Finish();
  if (int64_level_map_writer_->Size() > 0) {
    reader->int64_level_map_reader_.reset(int64_level_map_writer_->
                                          MoveToReader().release());
    reader->int64_level_map_itr_ =
        reader->int64_level_map_reader_->GetIterator();
  }

  return std::unique_ptr<IndexReader>(reader);
}

void VectorIndexBuilder::ProcessData(const FeatureData& data) {
  if (!data.HasFeature(feature_family_)) {
    this->WriteLevelId(-1);      // means no this feature family
    // we still need to process the potential scaling parameter
  } else if (data.HasStringValues(feature_family_)) {
    assert(int64_level_to_id_.empty());
    const std::string& level = data.FeatureValue(feature_family_);
    this->WriteLevelId(GetStrLevelId(level));
  } else {
    int64 level = data.FeatureValueInt64(feature_family_);
    this->WriteLevelId(GetInt64LevelId(level));
  }

  // TODO(kuehnelf): improve upon this hack
  double scaling = data.FeatureScaling(feature_family_, -1e200);
  if (scaling > -1e200) this->MayWriteScaling(scaling);
}

int VectorIndexBuilder::GetFeatureLevelId(const FeatureData& data) const {
  if (data.HasFeature(feature_family_)) {
    if (data.HasStringValues(feature_family_)) {
      const auto iter = str_level_to_id_.find(
          data.FeatureValue(feature_family_));
      if (iter != str_level_to_id_.end()) {
        return iter->second;
      }
    } else {
      const auto iter = int64_level_to_id_.find(
          data.FeatureValueInt64(feature_family_));
      if (iter != int64_level_to_id_.end()) {
        return iter->second;
      }
    }
  }
  return -1;
}

}  // namespace emre
