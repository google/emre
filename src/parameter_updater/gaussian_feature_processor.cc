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

#include "base/logging.h"

namespace emre {

using util::ArraySlice;
using util::MutableArraySlice;

void GaussianFeatureProcessor::AddToPrediction(
    IndexReader* index,
    ArraySlice<double> coefficients,
    MutableArraySlice<double> pred_values) {
  auto p_value_iter = pred_values.begin();
  auto level_iter = index->GetLevelIterator();
  auto scaling_iter = index->GetScalingIterator();
  for (; !level_iter.Done(); ++p_value_iter) {
    const int level_index = level_iter.Next();
    const double scl = scaling_iter.Next();
    if (level_index >= 0) {
      *p_value_iter += coefficients[level_index] * scl;
    }
  }
}

void GaussianFeatureProcessor::GetStatsForUpdateGaussGaussImpl(
    IndexReader* index,
    ArraySlice<double> inverse_variance,
    ArraySlice<double> coefficients,
    ArraySlice<double> pred_values,
    MutableArraySlice<double> residual_error) {
  CHECK_NOTNULL(index);
  const int num_levels = index->GetNumLevels();
  const int num_obs = index->GetNumObservations();
  CHECK_EQ(num_obs, inverse_variance.size());
  CHECK_GE(residual_error.size(), num_levels);
  auto p_value_iter = pred_values.begin();
  auto inv_var_iter = inverse_variance.begin();
  auto level_itr = index->GetLevelIterator();
  auto scaling_itr = index->GetScalingIterator();

  for (int j = 0; j < num_obs; ++j, ++p_value_iter, ++inv_var_iter) {
    const int level_index = level_itr.Next();
    const double scl = scaling_itr.Next();
    const double inv_var = *inv_var_iter;

    if (level_index >= 0) {
      double error = *p_value_iter - coefficients[level_index];
      residual_error[level_index] -= error * inv_var * scl;
    }
  }
}

void GaussianFeatureProcessor::UpdatePredictions(
    IndexReader* index,
    ArraySlice<double> coefficient_changes,
    MutableArraySlice<double> pred_values)  {
  const int n_obs = index->GetNumObservations();
  CHECK_GE(pred_values.size(), n_obs);
  auto p_value_iter = pred_values.begin();
  auto level_iter = index->GetLevelIterator();
  for (int i = 0; i < n_obs; ++i, ++p_value_iter) {
    const int level_index = level_iter.Next();
    if (level_index >= 0) {
      (*p_value_iter) += coefficient_changes[level_index];
    }
  }
}

}  // namespace emre
