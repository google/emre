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

// The usual random effects notation involves a double index, so we will start
// by describing the computations that way and then switch to a single index
// notation that is more similar to the actual data structures used.
//
// The model below reflects the computaton we need to perform to update a single
// feature family's random effects after conditioning on all other model
// parameters (e.g. random effects for other feature families).
//
// For the "Gaussian-Gaussian" model (Gaussian obsevation and Gaussian priors):
//   y_{kj} ~ b_{kj} + x_j * v_{kj} + eps_{jk} * s_{kj}
// where
//   - j = 1, 2, ..., r indexes the random effect
//   - b_kj is a known offset
//   - v_{kj} is a known scaling
//   - s_{kj}^2 is a known noise variance and
//   - x_j ~ N(0, \tau^-1) and eps_{kj} ~ N(0, 1) are the random effect and
//     noise distributions respectively.
// To compute the posterior distribution of x_j we need to calculate
//   1) s(1, j) := \sum_k v_{kj} (y_{kj} - b_{kj}) s_{kj}^-2
//   2) s(2, j) := \sum_k v_{kj}^2 s_{kj}^-2
// These statistics are stored in StatsForUpdate.first and StatsForUpdate.second
// respectively. From these statistics we can find the posterior Gaussian
// distribution for the random effect x_j.
//
// Now let's switch to a "single index" version that more accurately reflects
// the way the data is stored in the implementation below:
//    y_k ~ N(b_k + v_k*x_{I(k)}, s_k^-2)
//    x_j ~ N(0, \tau^-1)
//    where y_k, b_k, s_k, \tau known/fixed and I(k) \in {1,2,...,r} maps the
//    an observation index to random effect index
// In practice there are other features and random effects but we condition on
// their values/estimates and fold them into the term b_k above.
//
// In order to draw x_j from its posterior we need to calculate the following
// quantities for each index j=1,...,r of the random effect:
//   1) s(1, j) := \sum_{k:I(k)=j} v_k (y_k - b_k) s_k^-2
//   2) s(2, j) := \sum_{k:I(k)=j} v_k^2 s_k^-2

#ifndef EMRE_PARAMETER_UPDATER_GAUSSIAN_FEATURE_PROCESSOR_H_  // NOLINT
#define EMRE_PARAMETER_UPDATER_GAUSSIAN_FEATURE_PROCESSOR_H_

#include "update_processor.h"  // NOLINT

namespace emre {

class GaussianFeatureProcessor : public UpdateProcessor {
 public:
  void AddToPrediction(IndexReader* index,
                       util::ArraySlice<double> coefficients,
                       util::MutableArraySlice<double> pred_values) override;

  void GetStatsForUpdate(
      IndexReader* index,
      util::ArraySlice<double> inverse_variance,
      util::ArraySlice<double> coefficients,
      util::ArraySlice<double> pred_values,
      util::MutableArraySlice<double> level_predicted_values) override {
    GetStatsForUpdateGaussGaussImpl(
        index, inverse_variance, coefficients, pred_values,
        level_predicted_values);
  }

  void UpdatePredictions(IndexReader* index,
                         util::ArraySlice<double> coefficient_changes,
                         util::MutableArraySlice<double> pred_values) override;

 private:
  static void GetStatsForUpdateGaussGaussImpl(
      IndexReader* index,
      util::ArraySlice<double> inverse_variance,
      util::ArraySlice<double> coefficients,
      util::ArraySlice<double> pred_values,
      util::MutableArraySlice<double> residual_error);
};

}  // namespace emre

#endif  // EMRE_PARAMETER_UPDATER_GAUSSIAN_FEATURE_PROCESSOR_H_  // NOLINT
