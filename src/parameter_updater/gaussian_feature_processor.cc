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

#include "gaussian_feature_processor.h"  // NOLINT

#include "contentads/analysis/caa/search_plus/regmh/emre/src/base/logging.h"

namespace emre {

using util::ArraySlice;
using util::MutableArraySlice;

void GaussianFeatureProcessor::AddToPrediction(
    IndexReader* index,
    ArraySlice<double> coefficients,
    MutableArraySlice<double> p_events) {
  auto p_events_iter = p_events.begin();
  auto level_iter = index->GetLevelIterator();
  auto scaling_iter = index->GetScalingIterator();
  for (; !level_iter.Done(); ++p_events_iter) {
    const int level_index = level_iter.Next();
    const double scl = scaling_iter.Next();
    if (level_index >= 0) {
      *p_events_iter += coefficients[level_index] * scl;
    }
  }
}

void GaussianFeatureProcessor::GetStatsForUpdateGaussGaussImpl(
    IndexReader* index,
    ArraySlice<double> offset,
    ArraySlice<double> coefficients,
    ArraySlice<double> p_events,
    MutableArraySlice<double> scaled_mean) {
  CHECK_NOTNULL(index);
  const int num_levels = index->GetNumLevels();
  const int num_obs = index->GetNumObservations();
  CHECK_EQ(num_obs, offset.size());
  CHECK_GE(scaled_mean.size(), num_levels);
  auto p_events_iter = p_events.begin();
  auto offset_iter = offset.begin();
  auto level_itr = index->GetLevelIterator();
  auto scaling_itr = index->GetScalingIterator();

  for (int j = 0; j < num_obs; ++j, ++p_events_iter, ++offset_iter) {
    const int level_index = level_itr.Next();
    const double scl = scaling_itr.Next();
    const double offset = *offset_iter;

    if (level_index >= 0) {
      double pred_events = *p_events_iter - coefficients[level_index];
      scaled_mean[level_index] -= pred_events * offset * scl;
    }
  }
}

void GaussianFeatureProcessor::UpdatePredictions(
    IndexReader* index,
    ArraySlice<double> coefficient_changes,
    MutableArraySlice<double> p_events)  {
  const int n_obs = index->GetNumObservations();
  CHECK_GE(p_events.size(), n_obs);
  auto p_events_itr = p_events.begin();
  auto level_iter = index->GetLevelIterator();
  for (int i = 0; i < n_obs; ++i, ++p_events_itr) {
    const int level_index = level_iter.Next();
    if (level_index >= 0) {
      (*p_events_itr) += coefficient_changes[level_index];
    }
  }
}

}  // namespace emre
