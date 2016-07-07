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

// IndexReader and IndexBuilder
// * reads/writes feature levels and scalings of a specific feature family
//   for all observations.
// * are interfaces for parameter updaters.
// * The "back-end" storage for these indexers are undefined, can be memory or
//   disk based.

#ifndef EMRE_INDEXERS_INDEXER_H_  // NOLINT
#define EMRE_INDEXERS_INDEXER_H_

#include <memory>
#include <vector>

#include "feature_index.h"  // NOLINT
#include "vector_storage.h"  // NOLINT
#include "base/integral_types.h"

namespace emre {

class IndexReader {
 public:
  virtual ~IndexReader() {}

  // The feature family this IndexReader handles.
  virtual const string GetFeatureFamily() const = 0;

  // Num of levels of the feature family.
  virtual int GetNumLevels() const = 0;

  // Num of observations.
  virtual int GetNumObservations() const = 0;

  // Get level iterator.
  // Note: This call invalidates all previous iterators.
  virtual VectorReader<int32>::Iterator GetLevelIterator() = 0;

  // Get scaling iterator.
  // Note: This call invalidates all previous iterators.
  virtual VectorReader<double>::Iterator GetScalingIterator() = 0;

  // FillFeatureLevel is based on the VectorReader interface,
  // it reads through its elements sequentially, hence we need to
  // reset it with 'ResetFeatureLevel' before actual use.
  virtual void ResetFeatureLevel() = 0;
  virtual void FillFeatureLevel(FeatureData* data) = 0;
};

// IndexReader whose all scalings equal 1.0
class IndexReaderWithDefaultScaling : public IndexReader {
 public:
  explicit IndexReaderWithDefaultScaling(int num_rows)
      : const_scaling_reader_(1.0, num_rows) {}

  VectorReader<double>::Iterator GetScalingIterator() override {
    return const_scaling_reader_.GetIterator();
  }

 private:
  ConstVectorReader<double> const_scaling_reader_;
};

class IndexBuilder {
 public:
  IndexBuilder() {}
  virtual ~IndexBuilder() {}

  // The feature family that this IndexWriter handles.
  virtual const string GetFeatureFamily() const = 0;

  // Process one observation.
  virtual void ProcessData(const FeatureData& data) = 0;

  // Move to IndexReader. Note: Do not use this object after move.
  virtual std::unique_ptr<IndexReader> MoveToReader() = 0;

  // Get feature level ID of the observation.
  virtual int GetFeatureLevelId(const FeatureData& data) const = 0;
};

}  // namespace emre

#endif  // EMRE_INDEXERS_INDEXER_H_  // NOLINT
