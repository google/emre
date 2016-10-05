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

// A BlockRelaxation object updates the weights (i.e. random effects) for an
// entire feature family based on summary statistics of the dataset.  This
// update is usually a coordinate ascent update or a Gibbs sampling step.
//
// In the Gamma-Poisson regression the summary statistics would be the
// 'predicted-conversions' which is based on all the _other_ feature families
// in the model (those not being updated) and the conversion count.  These
// statistics are added up per level in passed in through the a vector of
// StatsForUpdate.
//
// A second function of the class is to compute a summary of the update step
// which, for Gibbs sampling, is the mean and variance of the conditional
// distribution.  This is done in the function VectorUpdateSampleStatistics
// and the resulting means and variances are saved to a vector of
// 'SampleStats' objects.

#ifndef EMRE_RANEF_UPDATER_BLOCK_RELAXATION_H_  // NOLINT
#define EMRE_RANEF_UPDATER_BLOCK_RELAXATION_H_

#include "parameter_updater/update_processor.h"
#include "training_data.pb.h"
#include "util/arrayslice.h"
#include "util/distribution.h"

namespace emre {

// TODO(kuehnelf): change name to CoefficientUpdater
class BlockRelaxation {
 public:
  // TODO(kuehnelf): find a better place
  struct MeanVar {
    double mean;
    double var;
  };

  struct UpdateParameters {
    UpdateParameters() {
      num_levels = 0;
    }

    explicit UpdateParameters(util::ArraySlice<double> predictions) {
      predicted = predictions;
      num_levels = predicted.size();
    }

    UpdateParameters(util::ArraySlice<double> predictions,
                     util::ArraySlice<double> auxiliary_data) {
      predicted = predictions;
      auxiliary = auxiliary_data;
      num_levels = predicted.size();
    }

    int num_levels;
    util::ArraySlice<double> predicted;  // size is equivalent to num_levels
    util::ArraySlice<double> auxiliary;
    util::ArraySlice<double> scores;

    // Updater-specific stats. For example, continous poisson updater uses
    // this field to store params which are only useful to itself.
    UpdateProcessor::SupplementalStats supplemental_stats;
  };

  BlockRelaxation() : distn_(nullptr) {}
  virtual ~BlockRelaxation() {}

  // This computes parameter (fixed effect or random effect) updates using the
  // prior distribution specified in 'prior' and the statistics stored in
  // 'UpdateParameters::stats'.  The nature of 'stats' depends on the
  // distribution being modeled e.g. gaussian, poisson, or exponential.
  virtual void UpdateRanefs(
      const FeatureFamilyPrior& prior,
      const UpdateParameters& update_parameters,
      util::MutableArraySlice<double> r) = 0;

  void SetRng(util::random::Distribution* rng) { distn_ = rng; }

  util::random::Distribution* GetRng() { return distn_; }

 protected:
  util::random::Distribution* distn_;
};

}  // namespace emre

#endif  // EMRE_RANEF_UPDATER_BLOCK_RELAXATION_H_  // NOLINT
