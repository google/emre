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

#include "gaussian_block_relaxation.h"  // NOLINT

#include <math.h>

#include "contentads/analysis/caa/search_plus/regmh/emre/src/util/distribution.h"

namespace emre {
namespace gaussian {

using util::MutableArraySlice;

void GaussianOptimizer::UpdateRanefs(
    const FeatureFamilyPrior& prior,
    const BlockRelaxation::UpdateParameters& update_parameters,
    MutableArraySlice<double> r) {
  const int num_levels = update_parameters.num_levels;
  CHECK_LE(num_levels, r.size());
  const double prior_mean = prior.mean();
  const double prior_invvar = prior.inverse_variance();

  auto mean_iter = update_parameters.predicted.begin();
  auto invvar_iter = update_parameters.auxiliary.begin();
  auto ret_iter = r.begin();
  for (int i = 0; i < num_levels; ++i, ++invvar_iter, ++mean_iter, ++ret_iter) {
    *ret_iter = CalculatePosteriorMeanVariance(
        prior_mean, prior_invvar, *mean_iter, *invvar_iter).mean;
  }
}

void GaussianGibbsSampler::UpdateRanefs(
    const FeatureFamilyPrior& prior,
    const BlockRelaxation::UpdateParameters& update_parameters,
    MutableArraySlice<double> r) {
  const int num_levels = update_parameters.num_levels;
  const double prior_mean = prior.mean();
  const double prior_invvar = prior.inverse_variance();

  auto* rng = this->GetRng();
  auto mean_iter = update_parameters.predicted.begin();
  auto invvar_iter = update_parameters.auxiliary.begin();
  auto ret_iter = r.begin();
  for (int i = 0; i < num_levels; ++i, ++invvar_iter, ++mean_iter, ++ret_iter) {
    *ret_iter = DrawPosteriorSample(prior_mean, prior_invvar,
                                    *mean_iter, *invvar_iter, rng);
  }
}

}  // namespace gaussian
}  // namespace emre
