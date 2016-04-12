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

#ifndef EMRE_RANEF_UPDATER_GAUSSIAN_BLOCK_RELAXATION_H_  // NOLINT
#define EMRE_RANEF_UPDATER_GAUSSIAN_BLOCK_RELAXATION_H_

#include "block_relaxation.h"  // NOLINT

namespace emre {
namespace gaussian {

// Does a coordinate update for a parameter in a gaussian regression.
class GaussianOptimizer : public BlockRelaxation {
 public:
  void UpdateRanefs(const FeatureFamilyPrior& prior,
                    const UpdateParameters& update_parameters,
                    util::MutableArraySlice<double> r) override;
};

class GaussianGibbsSampler : public GaussianOptimizer {
 public:
  void UpdateRanefs(const FeatureFamilyPrior& prior,
                    const UpdateParameters& update_parameters,
                    util::MutableArraySlice<double> r) override;
};

inline BlockRelaxation::MeanVar CalculatePosteriorMeanVariance(
    double prior_mean, double prior_invvar,
    double stats_mean, double stats_invvar) {
  double post_invvar = prior_invvar + stats_invvar;
  return BlockRelaxation::MeanVar{
      (stats_mean + prior_mean * prior_invvar) / post_invvar,
      1.0 / post_invvar};
}

inline double DrawPosteriorSample(
    double prior_mean, double prior_invvar,
    double stats_mean, double stats_invvar,
    util::random::Distribution* rng) {
  auto post = CalculatePosteriorMeanVariance(
      prior_mean, prior_invvar, stats_mean, stats_invvar);
  return post.mean + rng->RandGaussian(sqrt(post.var));
}

}  // namespace gaussian
}  // namespace emre

#endif  // _EMRE_RANEF_UPDATER_GAUSSIAN_BLOCK_RELAXATION_H_  // NOLINT

