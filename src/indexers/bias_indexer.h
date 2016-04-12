// Copyright 2010-2016 Google
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

#ifndef EMRE_INDEXERS_BIAS_INDEXER_H_  // NOLINT
#define EMRE_INDEXERS_BIAS_INDEXER_H_

#include "indexer.h"  // NOLINT

namespace emre {

class BiasIndexReader : public IndexReaderWithDefaultScaling {
 public:
  explicit BiasIndexReader(int num_observations)
      : IndexReaderWithDefaultScaling(num_observations),
        num_observations_(num_observations),
        const_level_reader_(0, num_observations) {}
  virtual ~BiasIndexReader() {}

  int GetNumLevels() const override { return 1; }

  int GetNumObservations() const override { return num_observations_; }

  const string GetFeatureFamily() const override;

  VectorReader<int32>::Iterator GetLevelIterator() override {
    return const_level_reader_.GetIterator();
  }

  void ResetFeatureLevel() override {}
  void FillFeatureLevel(FeatureData* data) override;

  static const char kFeatureFamily[];

 private:
  int num_observations_;
  ConstVectorReader<int32> const_level_reader_;
};

}  // namespace emre

#endif  // EMRE_INDEXERS_BIAS_INDEXER_H_  // NOLINT
