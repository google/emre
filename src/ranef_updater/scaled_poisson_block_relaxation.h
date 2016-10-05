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

#ifndef EMRE_RANEF_UPDATER_SCALED_POISSON_BLOCK_RELAXATION_H_  // NOLINT
#define EMRE_RANEF_UPDATER_SCALED_POISSON_BLOCK_RELAXATION_H_

#include <vector>

#include "poisson_block_relaxation.h"  // NOLINT

namespace emre {

class ScaledPoissonOptimizer : public GammaPoissonOptimizer {
 public:
  void UpdateRanefs(const FeatureFamilyPrior& prior,
                    const UpdateParameters& update_parameters,
                    util::MutableArraySlice<double> r) override;
};

class ScaledPoissonGibbsSampler : public ScaledPoissonOptimizer {
 public:
  ScaledPoissonGibbsSampler();

  void UpdateRanefs(const FeatureFamilyPrior& prior,
                    const UpdateParameters& update_parameters,
                    util::MutableArraySlice<double> r) override;

 private:
  // these values are hard coded for now!
  const int num_steps_per_iteration_;  //  number MH proposal steps.
  std::vector<double> proposal_sds_;
  std::vector<std::pair<int, int>> acceptance_counts_;
};

}  // namespace emre

#endif  // EMRE_RANEF_UPDATER_SCALED_POISSON_BLOCK_RELAXATION_H_  // NOLINT
