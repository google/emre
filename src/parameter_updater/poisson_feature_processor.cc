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

#include "poisson_feature_processor.h"  // NOLINT

#include "base/logging.h"
#include "indexers/memory_vector.h"
#include "scaled_feature_util.h"  // NOLINT

namespace emre {

using util::ArraySlice;
using util::MutableArraySlice;

void PoissonFeatureProcessor::AddToPrediction(
    IndexReader* index,
    ArraySlice<double> coefficients,
    MutableArraySlice<double> p_events) {
  auto p_events_iter = p_events.begin();
  auto level_iter = index->GetLevelIterator();
  for (; !level_iter.Done(); ++p_events_iter) {
    const int level_index = level_iter.Next();
    if (level_index >= 0) {
      *p_events_iter *= coefficients[level_index];
    }
  }
}

void PoissonFeatureProcessor::GetStatsForUpdatePoissonGammaImpl(
    IndexReader* index,
    ArraySlice<double> coefficients,
    ArraySlice<double> p_events,
    MutableArraySlice<double> expected_events) {
  const int num_obs = index->GetNumObservations();
  const int num_levels = index->GetNumLevels();
  CHECK_GE(expected_events.size(), num_levels);
  auto p_events_iter = p_events.begin();
  auto level_iter = index->GetLevelIterator();

  for (int i = 0; i < num_obs; ++i, ++p_events_iter) {
    const int level_index = level_iter.Next();
    if (level_index >= 0) {
      double pred_events = *p_events_iter / coefficients[level_index];
      expected_events[level_index] += pred_events;
    }
  }
}

void PoissonFeatureProcessor::UpdatePredictions(
    IndexReader* index,
    ArraySlice<double> coefficient_changes,
    MutableArraySlice<double> p_events) {
  auto p_events_iter = p_events.begin();
  auto level_iter = index->GetLevelIterator();
  for (; !level_iter.Done(); ++p_events_iter) {
    const int level_index = level_iter.Next();
    if (level_index >= 0) {
      (*p_events_iter) *= coefficient_changes[level_index];
    }
  }
}

PoissonScaledFeatureProcessor::PoissonScaledFeatureProcessor(
    IndexReader* index) : cached_index_(index) {
  std::unique_ptr<VectorBuilder<int>> level_scaling_posn_builder(
      new MemoryVectorBuilder<int>());
  ScaledFeatureUtil::MakeLevelScalingMapping(
      index, &level_posn_size_, level_scaling_posn_builder.get(),
      &aggregate_scaling_);

  level_scaling_posn_ = level_scaling_posn_builder->MoveToReader();
}

void PoissonScaledFeatureProcessor::PrepareUpdater(SupplementalStats* stats) {
  CHECK_NOTNULL(stats);
  stats->aggregate_scaling = ArraySlice<double>(aggregate_scaling_);
  stats->level_posn_size = ArraySlice<pair<int, int>>(level_posn_size_);
}

void PoissonScaledFeatureProcessor::GetStatsForUpdateScaledPoissonImpl(
    IndexReader* index,
    ArraySlice<double> coefficients,
    ArraySlice<double> p_events,
    MutableArraySlice<double> prediction) {
  CHECK_EQ(index, cached_index_);
  const int output_size = aggregate_scaling_.size();
  CHECK_GE(prediction.size(), output_size);

  ScaledFeatureUtil::GetPredictionForPoissonUpdate(
      index, level_scaling_posn_.get(), coefficients, p_events, prediction);
}

void PoissonScaledFeatureProcessor::AddToPrediction(
    IndexReader* index,
    ArraySlice<double> coefficients,
    MutableArraySlice<double> p_events) {
  CHECK_EQ(index, cached_index_);
  auto p_events_iter = p_events.begin();
  auto level_iter = index->GetLevelIterator();
  auto scaling_iter = index->GetScalingIterator();
  for (; !level_iter.Done(); ++p_events_iter) {
    const int level_index = level_iter.Next();
    const double scaling = scaling_iter.Next();
    if (level_index >= 0) {
      (*p_events_iter) *= exp(scaling * coefficients[level_index]);
    }
  }
}

void PoissonScaledFeatureProcessor::UpdatePredictions(
    IndexReader* index,
    ArraySlice<double> coefficient_changes,
    MutableArraySlice<double> p_events) {
  CHECK_EQ(index, cached_index_);
  auto p_events_iter = p_events.begin();
  auto level_iter = index->GetLevelIterator();
  auto scaling_iter = index->GetScalingIterator();
  for (; !level_iter.Done(); ++p_events_iter) {
    const int level_index = level_iter.Next();
    const double scaling = scaling_iter.Next();
    if (level_index >= 0) {
      (*p_events_iter) *= exp(scaling * coefficient_changes[level_index]);
    }
  }
}

PoissonLogNormalFeatureProcessor::PoissonLogNormalFeatureProcessor(
    IndexReader* index) {
  SetCachedIndex(index);
  // TODO(kuehnelf): This is a simple change to implement log-normal priors,
  // and utilizes the scaled poisson infrastructure. However, the API's nor
  // the data structures are efficient.
  int num_levels = index->GetNumLevels();
  trivial_aggregate_scaling_.assign(num_levels, 1.0);
  trivial_level_posn_size_.assign(num_levels, pair<int, int>(0, 1));
  for (int i = 0; i < num_levels; ++i) {
    trivial_level_posn_size_[i].first = i;
  }
}

void PoissonLogNormalFeatureProcessor::GetStatsForUpdatePoissonLogNormalImpl(
    IndexReader* index,
    ArraySlice<double> coefficients,
    ArraySlice<double> p_events,
    MutableArraySlice<double> expected_events) {
  const int num_obs = index->GetNumObservations();
  const int num_levels = index->GetNumLevels();
  CHECK_GE(expected_events.size(), num_levels);
  auto p_events_iter = p_events.begin();
  auto level_iter = index->GetLevelIterator();

  for (int i = 0; i < num_obs; ++i, ++p_events_iter) {
    const int level_index = level_iter.Next();
    if (level_index >= 0) {
      double pred_events = *p_events_iter * exp(-coefficients[level_index]);
      expected_events[level_index] += pred_events;
    }
  }
}

void PoissonLogNormalFeatureProcessor::PrepareUpdater(
    SupplementalStats* stats) {
  CHECK_NOTNULL(stats);
  stats->aggregate_scaling = trivial_aggregate_scaling_;
  stats->level_posn_size = trivial_level_posn_size_;
}

}  // namespace emre
