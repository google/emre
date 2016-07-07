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

// A 'PriorUpdater' object fits or Gibbs samples the prior parameters for a
// feature family.  The update may depend on either the summary statistics for
// each level (passed in through a vector of StatsForUpdate) or the current
// weights/scores/random-effects passed in through a double vector.  Updates
// typically depend on just one or the other of these two inputs.

#ifndef EMRE_PRIOR_UPDATER_PRIOR_UPDATER_H_  // NOLINT
#define EMRE_PRIOR_UPDATER_PRIOR_UPDATER_H_

#include "training_data.proto.h"
#include "util/arrayslice.h"
#include "util/distribution.h"

namespace emre {

class PriorUpdater {
 public:
  virtual ~PriorUpdater() {}

  // 'scores' should have one element per feature level.  'stats' may be
  // longer than 'scores' but only the first 'scores.size()' elements will
  // be used.
  virtual void UpdateVariance(
      util::ArraySlice<double> ancillary,
      util::ArraySlice<double> prediction,  // pass in the smoothed predictions
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) = 0;

  virtual void SetProtoFromPrior(FeatureFamilyPrior* pb) const = 0;
  virtual void SetPriorFromProto(const FeatureFamilyPrior& pb) = 0;

  // This structure holds parameters to the optimization function that updates
  // the prior.  The 'num_iterations' field is reused in MCMC updates to
  // indicate the number of Metropolis Hastings (or other) steps.
  struct PriorOptimConfig {
   public:
    PriorOptimConfig() : grid_size(10), num_iterations(4), frequency(0) {}
    int grid_size;
    int num_iterations;
    int frequency;
  };

  const PriorOptimConfig& GetPriorOptimConfig() const { return optim_config_; }
  void SetPriorOptimConfig(const PriorOptimConfig& config) {
    optim_config_ = config;
  }

 protected:
  PriorOptimConfig optim_config_;
};

// as the name says, it simply does nothing to the prior!
class DoesNothingPriorUpdater : public PriorUpdater {
 public:
  DoesNothingPriorUpdater() {}
  ~DoesNothingPriorUpdater() final {}

  void UpdateVariance(
      util::ArraySlice<double> ancillary,
      util::ArraySlice<double> prediction,
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) override {};

  void SetProtoFromPrior(FeatureFamilyPrior* pb) const override {
    *pb = ffp_;  // copy into buffer
  }
  void SetPriorFromProto(const FeatureFamilyPrior& pb) override {
    ffp_ = pb;
  }

 private:
  FeatureFamilyPrior ffp_;
};

}  // namespace emre

#endif  // EMRE_PRIOR_UPDATER_PRIOR_UPDATER_H_  // NOLINT
