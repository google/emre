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

#ifndef EMRE_RANEF_UPDATER_POISSON_BLOCK_RELAXATION_H_  // NOLINT
#define EMRE_RANEF_UPDATER_POISSON_BLOCK_RELAXATION_H_

#include "block_relaxation.h"  // NOLINT

namespace emre {

// Does a coordinate update for a parameter in a gamma-poisson regression.
class GammaPoissonOptimizer : public BlockRelaxation {
 public:
  void UpdateRanefs(const FeatureFamilyPrior& prior,
                    const UpdateParameters& update_parameters,
                    util::MutableArraySlice<double> r) override;
};

class GammaPoissonGibbsSampler : public GammaPoissonOptimizer {
 public:
  void UpdateRanefs(const FeatureFamilyPrior& prior,
                    const UpdateParameters& update_parameters,
                    util::MutableArraySlice<double> r) override;
};

}  // namespace emre

#endif  // EMRE_RANEF_UPDATER_POISSON_BLOCK_RELAXATION_H_  // NOLINT
