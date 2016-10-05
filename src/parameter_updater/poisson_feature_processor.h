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

#ifndef EMRE_PARAMETER_UPDATER_POISSON_FEATURE_PROCESSOR_H_  // NOLINT
#define EMRE_PARAMETER_UPDATER_POISSON_FEATURE_PROCESSOR_H_

#include "update_processor.h"  // NOLINT

namespace emre {

class PoissonFeatureProcessor : public UpdateProcessor {
 public:
  void AddToPrediction(IndexReader* index,
                       util::ArraySlice<double> coefficients,
                       util::MutableArraySlice<double> p_events) override;

  void GetStatsForUpdate(
      IndexReader* index,
      util::ArraySlice<double> offsets,
      util::ArraySlice<double> coefficients,
      util::ArraySlice<double> p_events,
      util::MutableArraySlice<double> level_predicted_events) override {
    GetStatsForUpdatePoissonGammaImpl(
        index, coefficients, p_events, level_predicted_events);
  }

  void UpdatePredictions(IndexReader* index,
                         util::ArraySlice<double> score_changes_buffer,
                         util::MutableArraySlice<double> p_events) override;

 private:
  static void GetStatsForUpdatePoissonGammaImpl(
      IndexReader* index,
      util::ArraySlice<double> coefficients,
      util::ArraySlice<double> p_events,
      util::MutableArraySlice<double> expected_events);
};

class PoissonScaledFeatureProcessor : public PoissonFeatureProcessor {
 public:
  PoissonScaledFeatureProcessor() {}
  explicit PoissonScaledFeatureProcessor(IndexReader* index);

  int GetStatsSize(int num_levels) override {
    return aggregate_scaling_.size();
  }

  void AddToPrediction(IndexReader* index,
                       util::ArraySlice<double> coefficients,
                       util::MutableArraySlice<double> p_events) override;

  void GetStatsForUpdate(
      IndexReader* index,
      util::ArraySlice<double> offsets,
      util::ArraySlice<double> coefficients,
      util::ArraySlice<double> p_events,
      util::MutableArraySlice<double> level_predicted_events) override {
    this->GetStatsForUpdateScaledPoissonImpl(
        index, coefficients, p_events, level_predicted_events);
  }

  void UpdatePredictions(IndexReader* index,
                         util::ArraySlice<double> score_changes_buffer,
                         util::MutableArraySlice<double> p_events) override;

  void PrepareUpdater(SupplementalStats* stats) override;

 protected:
  void SetCachedIndex(IndexReader* index) { cached_index_ = index; }

 private:
  void GetStatsForUpdateScaledPoissonImpl(
      IndexReader* index,
      util::ArraySlice<double> coefficients,
      util::ArraySlice<double> p_events,
      util::MutableArraySlice<double> expected_events);

  // We precompute these mappings which are used in GetStatsForUpdate.  If the
  // pointer 'index' is the same as the one saved in 'cached_index_' then we
  // assume the mappings are already computed in the vectors below.
  IndexReader* cached_index_;  // Not owned

  // The following hold mapping used to do an aggregation of the data.  See
  // scaled_feature_util.h and the design doc it links to for details.
  std::vector<std::pair<int, int>> level_posn_size_;
  std::unique_ptr<VectorReader<int>> level_scaling_posn_;
  std::vector<double> aggregate_scaling_;
};

class PoissonLogNormalFeatureProcessor : public PoissonScaledFeatureProcessor {
 public:
  explicit PoissonLogNormalFeatureProcessor(IndexReader* index);

  int GetStatsSize(int num_levels) override { return num_levels; }

  void GetStatsForUpdate(
      IndexReader* index,
      util::ArraySlice<double> offsets,
      util::ArraySlice<double> coefficients,
      util::ArraySlice<double> p_events,
      util::MutableArraySlice<double> level_predicted_events) override {
    GetStatsForUpdatePoissonLogNormalImpl(
        index, coefficients, p_events, level_predicted_events);
  }

  void PrepareUpdater(SupplementalStats* stats) override;

 private:
  static void GetStatsForUpdatePoissonLogNormalImpl(
      IndexReader* index,
      util::ArraySlice<double> coefficients,
      util::ArraySlice<double> p_events,
      util::MutableArraySlice<double> expected_events);

  // The following hold mapping used to do an aggregation of the data.  See
  // scaled_feature_util.h and the design doc it links to for details.
  std::vector<std::pair<int, int>> trivial_level_posn_size_;
  std::vector<double> trivial_aggregate_scaling_;
};

}  // namespace emre

#endif  // EMRE_PARAMETER_UPDATER_POISSON_FEATURE_PROCESSOR_H_  // NOLINT
