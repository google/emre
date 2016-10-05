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

// In the Gaussian model the two statistics we collect change meaning.  If we
// have observations x_1, x_2, ..., x_n modeled as
//   x_k ~ N(y_k, s_k^2)
//   y_k ~ N(mu, sigma^2)
// then StatsForUpdate = (x_k * s_k^-2, s_k^-2)
// and the posterior distribution is:
//    {y_k|x_k, sigma, s_k} ~ N(m, v^2)
// where v^2 = 1 / (sigma^-2 + s_k^-2)
// and m = (mu/sigma^2 + x_k/s_k^2) * v^2
//
// So all the formulas are simpler if we work with inverse variances e.g. s_k^-2
// and so the posterior mean and variance is written as
//   v^2 = 1 / (sigma^-2 + stats.second)
//   m = (mu/sigma^2 + stats.first) * v^2
//
// The marginal log likelihood is
//    0.5 * m * m * v^-2 - 0.5 * mu * mu * sigma^-2
//    - 0.5 * log(v^-2) + 0.5 * log(sigma^-2)
// and in the code we drop the factor of 0.5.
//
// A non-zero prior mean ('mu' above) is not standard.  The code below allows
// it to be input but does not currently fit 'mu' -- only the prior variance
// sigma^2 which is stored as sigma^-2 in prior_inverse_variance_

// Fits the Gaussian prior using the likelihood of sample random effects

#ifndef EMRE_PRIOR_UPDATER_GAUSSIAN_PRIOR_OPTIMIZE_H_  // NOLINT
#define EMRE_PRIOR_UPDATER_GAUSSIAN_PRIOR_OPTIMIZE_H_

#include "base/logging.h"
#include "gamma_prior_optimize.h"  // NOLINT

namespace emre {

class GaussianPriorSampleOptimize : public PriorUpdater {
 public:
  GaussianPriorSampleOptimize()
      : prior_mean_(0.0), prior_inverse_variance_(1.0) {}

  // 'stats' and 'rng' are not used.  rng may be null
  void UpdateVariance(
      util::ArraySlice<double> stats_invvar,  // ancillary
      util::ArraySlice<double> stats_error,  // prediction error
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) override;

  // Accesses and sets the fields 'mean' and 'inverse_variance'
  void SetProtoFromPrior(FeatureFamilyPrior* pb) const override;
  void SetPriorFromProto(const FeatureFamilyPrior& pb) override;

 protected:
  double GetPriorMean() const { return prior_mean_; }
  void SetPriorMean(double m) { prior_mean_ = m; }

  double GetPriorInverseVariance() const { return prior_inverse_variance_; }
  void SetPriorInverseVariance(double v) {
    CHECK_GT(v, 0.0);
    prior_inverse_variance_ = v;
  }

 private:
  int max_levels_for_update_;
  double prior_mean_;
  double prior_inverse_variance_;
};

// Fits the Gaussian prior using an integrated update which uses the
// likelihood after integrating out the random effect.
class GaussianPriorIntegratedOptimize : public GaussianPriorSampleOptimize {
 public:
  // 'rng' is not used and may be null
  void UpdateVariance(
      util::ArraySlice<double> stats_invvar,  // ancillary
      util::ArraySlice<double> stats_error,  // prediction error
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) override;
};

// Fits the Gaussian prior using an integrated update which uses the
// likelihood after integrating out the random effect.
class GaussianPriorGibbsSampler : public GaussianPriorSampleOptimize {
 public:
  GaussianPriorGibbsSampler() :
      num_steps_(0), num_accepted_(0), proposal_sd_(1.0) {}

  void SetProposalStdDev(double sd) { proposal_sd_ = sd; }
  double GetProposalStdDev() const { return proposal_sd_; }

  // 'stats' is not used and may be null
  void UpdateVariance(
      util::ArraySlice<double> stats_invvar,  // ancillary
      util::ArraySlice<double> stats_error,  // prediction error
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) override;

 protected:
  virtual void UpdateProposalStdDev();

 private:
  int num_steps_;
  int num_accepted_;
  double proposal_sd_;
};

}  // namespace emre

#endif  // EMRE_PRIOR_UPDATER_GAUSSIAN_PRIOR_OPTIMIZE_H_  // NOLINT
