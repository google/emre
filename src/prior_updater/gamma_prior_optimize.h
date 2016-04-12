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

#ifndef EMRE_PRIOR_UPDATER_GAMMA_PRIOR_OPTIMIZE_H_  // NOLINT
#define EMRE_PRIOR_UPDATER_GAMMA_PRIOR_OPTIMIZE_H_

#include <vector>

#include "contentads/analysis/caa/search_plus/regmh/emre/src/util/basic_types.h"
#include "prior_updater.h"  // NOLINT

namespace emre {

// This specializes PriorUpdater for a gamma-poisson model (where the prior
// distribution is gamma).
class GammaPoissonPriorUpdater : public PriorUpdater {
 public:
  GammaPoissonPriorUpdater() : alpha_(10.0), beta_(10.0) {}
  virtual ~GammaPoissonPriorUpdater() {}

  virtual void UpdateVariance(
      util::ArraySlice<double> stats_events,  // ancillary
      util::ArraySlice<double> stats_p_events,  // pass in the smoothed pred.
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) = 0;

  void SetProtoFromPrior(FeatureFamilyPrior* pb) const override;
  void SetPriorFromProto(const FeatureFamilyPrior& pb) override;

 public:
  void SetGammaPrior(double alpha, double beta) {
    alpha_ = alpha;
    beta_ = beta;
  }

  int GetMaxLevels() const {
    return max_levels_for_update_;
  }

  pair<double, double> GetGammaPrior() const {
    return pair<double, double>(alpha_, beta_);
  }

 private:
  double alpha_;
  double beta_;
  int max_levels_for_update_;
};

// Fits the gamma prior using an integrated update which uses the likelihood
// after integrating out the random effect
class GammaPoissonPriorIntegratedOptimize : public GammaPoissonPriorUpdater {
 public:
  void UpdateVariance(
      util::ArraySlice<double> stats_events,  // ancillary
      util::ArraySlice<double> stata_p_events,  // pass in smoothed predictions
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) override;
};

// Fits the gamma prior using a basic update which only depends on the sample
// random effects held in 'scores'
class GammaPoissonPriorSampleOptimize : public GammaPoissonPriorUpdater {
 public:
  void UpdateVariance(
      util::ArraySlice<double> stats_events,  // ancillary
      util::ArraySlice<double> stats_p_events,  // pass in smoothed predictions
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) override;
};

// Fits the gamma prior using a "Rao-Blackwellized" that integrates over
// all outcomes of the last Gibbs sampling step.
class GammaPoissonPriorRaoBlackwellizedOptimize
    : public GammaPoissonPriorUpdater {
 public:
  void UpdateVariance(
      util::ArraySlice<double> stats_events,  // ancillary
      util::ArraySlice<double> stats_p_events,  // pass in smoothed predictions
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) override;
};

// Gibbs samples the gamma prior using an integrated update which uses the
// likelihood after integrating out the random effect.  After performing
// this update, the random effects should also be sampled.  This two stage
// update would be equivalent to a block sampling step from the distribution
//  p(prior, ranefs | other-parameters, observed-poissons)
class GammaPoissonPriorIntegratedGibbsSampler
    : public GammaPoissonPriorUpdater {
 public:
  void UpdateVariance(
      util::ArraySlice<double> stats_events,  // ancillary
      util::ArraySlice<double> stats_p_events,  // pass in smoothed predictions
      util::ArraySlice<double> coefficients,
      util::random::Distribution* rng) override;
};

// RealFunction and RealFunctionInLogSpace are exposed for testing purposes.

class RealFunction : public LogLikelihodFunction<double> {
 public:
  virtual ~RealFunction() {}
  // Assumes that 'r' is non-NULL and has length at least that of 'x'
  virtual void Evaluate(util::ArraySlice<double> x,
                        util::MutableArraySlice<double> r) const = 0;
};

// Acts as this->Evaluate(y) = cback(exp(y)) where cback is a function passed
// in to the constructor.  This is used to change the parameterization of
// prior parameters when updating through MCMC.
class RealFunctionInLogSpace : public RealFunction {
 public:
  RealFunctionInLogSpace(RealFunction* cback, bool should_own);
  virtual ~RealFunctionInLogSpace();

  // Assumes that 'r' is non-NULL and has length at least that of 'x'
  void Evaluate(util::ArraySlice<double> x,
                util::MutableArraySlice<double> r) const override;

 private:
  RealFunction* cback_;
  const bool should_own_;
};

// FindLikelihoodRangeForMCMC and NonNegativeOptimize are exposed for testing
// purposes.

// This function is used in MCMC sampling of prior parameters.  It searches
// in the range
//    [x0 - max_log_scaling, x0 + max_log_scaling]
// (where x0 = log_initial_value) for an interval such that both end-points
// are no more than 'max_llik_delta' lower than cback(exp(initial_value)).
// in likelihood.
//
// Returns the width 'R', so if max_llik_delta = 2 then R * 0.5 would be a
// reasonable choice of proposal standard deviation in Metropolis Hastings.
//
// max_llik_delta should be non-negative and the function will check for this.

pair<double, double> FindLikelihoodRangeForMCMC(
    const RealFunction& cback, double initial_value, int grid_size,
    double max_llik_delta, const pair<double, double>& range);


// This function is exposed for testing purposes.
void NonNegativeOptimize(const RealFunction& cback,
                         double initial_value,
                         int grid_size, int num_zoom,
                         vector<double>* evaluation_points,
                         vector<double>* values,
                         int* max_index);


// EmreSampleParamLikelihood can be used by other priors as well e.g.
// mixture of gamma and spike+gamma priors
class EmreSampleParamLikelihood : public RealFunction {
 public:
  explicit EmreSampleParamLikelihood(util::ArraySlice<double> sample_params)
      : sample_params_(sample_params) {}

  virtual ~EmreSampleParamLikelihood() {}

  void Evaluate(util::ArraySlice<double> x,
                util::MutableArraySlice<double> r) const override;

 protected:
  const util::ArraySlice<double> GetSampleParams() const {
    return sample_params_;
  }

 private:
  const util::ArraySlice<double> sample_params_;
};

// EmreParamIntegratedLikelihood can be used by other priors as well e.g.
// mixture of gamma and spike+gamma priors
class EmreParamIntegratedLikelihood : public RealFunction {
 public:
  // Doesn't own predicted and ancillary,
  // only keeps pointers via ArraySlice
  EmreParamIntegratedLikelihood(
      util::ArraySlice<double> predicted,
      util::ArraySlice<double> ancillary)
      : predicted_(predicted), ancillary_(ancillary) {}

  virtual ~EmreParamIntegratedLikelihood() {}

  void Evaluate(util::ArraySlice<double> x,
                util::MutableArraySlice<double> r) const override;

 protected:
  int NumElts() const { return predicted_.size(); }

  static void FillLookupTable(
      util::ArraySlice<double> x,
      util::MutableArraySlice<double> r,
      int num_elts);

  util::ArraySlice<double> predicted_;
  util::ArraySlice<double> ancillary_;
};

}  // namespace emre

#endif  // EMRE_PRIOR_UPDATER_GAMMA_PRIOR_OPTIMIZE_H_  // NOLINT
