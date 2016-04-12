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

#include "poisson_block_relaxation.h"  // NOLINT

#include <math.h>

namespace emre {

using util::MutableArraySlice;

void GammaPoissonOptimizer::UpdateRanefs(
    const FeatureFamilyPrior& prior,
    const UpdateParameters& update_parameters,
    MutableArraySlice<double> r) {
  const int num_levels =  update_parameters.num_levels;
  CHECK_LE(num_levels, r.size());

  auto events = update_parameters.auxiliary;
  auto p_events = update_parameters.predicted;
  CHECK_EQ(events.size(), p_events.size());
  auto ret_iter = r.begin();
  double alpha = prior.inverse_variance();
  double beta = prior.inverse_variance();

  // Due to some optimizations in the algorithm is not convenient to return an
  // exact 0.0 since it will take more work elsewhere to avoid divide-by-zero
  // errors.
  const double kMinEstimate = 1e-9;

  if (alpha > 0.0 && beta > 0.0) {
    // To get to the mode of the gamma distribution we take
    // (alpha - 1 + events) / (beta + expected-events)
    // where expected-events is based on all other features in the model
    alpha -= 1.0;
    if (alpha > 0.0) {
      for (int i = 0; i < num_levels; ++i, ++ret_iter) {
        *ret_iter = (events[i] + alpha) / (p_events[i] + beta);
      }
    } else {
      for (int i = 0; i < num_levels; ++i, ++ret_iter) {
        *ret_iter = std::max(kMinEstimate,
                             (events[i] + alpha) / (p_events[i] + beta));
      }
    }
  } else {
    for (int i = 0; i < num_levels; ++i, ++ret_iter) {
      *ret_iter = events[i] / p_events[i];
    }
  }
}

void GammaPoissonGibbsSampler::UpdateRanefs(
    const FeatureFamilyPrior& prior,
    const UpdateParameters& update_parameters,
    MutableArraySlice<double> r) {
  const int num_levels = update_parameters.num_levels;
  auto events = update_parameters.auxiliary;
  auto p_events = update_parameters.predicted;
  CHECK_EQ(events.size(), p_events.size());
  auto ret_iter = r.begin();
  const double alpha = prior.inverse_variance();
  const double beta = prior.inverse_variance();
  auto* rng = this->GetRng();
  CHECK_NOTNULL(rng);

  for (int i = 0; i < num_levels; ++i, ++ret_iter) {
    // draw a random gamma with mean equal to
    // (alpha + events) / (beta + expected-events)
    // where expected-events is based on all other features in the model
    *ret_iter = rng->RandGamma(alpha + events[i], 1.0 / (beta + p_events[i]));
  }
}

}  // namespace emre
