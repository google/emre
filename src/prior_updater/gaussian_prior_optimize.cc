// Copyright 2012-2015 Google
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

#include <math.h>
#include <algorithm>

#include "gaussian_prior_optimize.h"  // NOLINT
#include "likelihoods.h"  // NOLINT
#include "metropolis_hastings.h"  // NOLINT

namespace emre {

using util::ArraySlice;
using util::MutableArraySlice;

namespace {

// This computes the marginal likelihood of the data after integrating
// over the random effect.  The 'Evaluate' function takes a vector of prior
// inverse-variance parameters.
class GaussianIntegratedLikelihoodForVarianceUpdate : public RealFunction {
 public:
  GaussianIntegratedLikelihoodForVarianceUpdate(
      ArraySlice<double> stats_invvar,
      ArraySlice<double> stats_error)
      : stats_error_(stats_error),
        stats_invvar_(stats_invvar),
        prior_mean_(0.0) {}

  void Evaluate(ArraySlice<double> x,
                MutableArraySlice<double> r) const override;
  void SetPriorMean(double m) { prior_mean_ = m; }

 private:
  ArraySlice<double> stats_error_;
  ArraySlice<double> stats_invvar_;
  int num_elts_;
  double prior_mean_;
};

}  // namespace

void GaussianPriorSampleOptimize::UpdateVariance(
    ArraySlice<double> stats_invvar,
    ArraySlice<double> stats_error,
    ArraySlice<double> coefficients,
    util::random::Distribution* rng) {
  CHECK_GT(coefficients.size(), 0);

  // We assume a mean-zero prior
  double sum_squares = 0.0;
  for (auto x : coefficients) {
    sum_squares += x * x;
  }
  this->SetPriorInverseVariance(
      static_cast<double>(coefficients.size()) / sum_squares);
}

void GaussianPriorSampleOptimize::SetProtoFromPrior(
    FeatureFamilyPrior* pb) const {
  pb->set_mean(prior_mean_);
  pb->set_inverse_variance(prior_inverse_variance_);
  pb->set_max_levels_for_update(max_levels_for_update_);
}

void GaussianPriorSampleOptimize::SetPriorFromProto(
    const FeatureFamilyPrior& pb) {
  CHECK_GT(pb.inverse_variance(), 0.0);
  this->SetPriorMean(pb.mean());
  this->SetPriorInverseVariance(pb.inverse_variance());
  max_levels_for_update_ = pb.max_levels_for_update();
}

void GaussianIntegratedLikelihoodForVarianceUpdate::Evaluate(
    ArraySlice<double> x, MutableArraySlice<double> r) const {
  // x[] holds in the inverse variance of the prior

  // We discard the factor of 0.5 that scales the log likelihood
  const int num_elts = stats_error_.size();
  for (int i = 0; i < x.size(); ++i) {
    const double prior_invvar = x[i];
    const double scaled_prior_mean = prior_invvar * prior_mean_;
    double marginal_lik =
        (log(prior_invvar) - prior_invvar * prior_mean_ * prior_mean_)
        * static_cast<double>(num_elts);
    for (int j = 0; j < num_elts; ++j) {
      // recall that StatsForUpdate = (x_k * s_k^-2, s_k^-2)  See header for
      // explanation of notation
      const double invvar = prior_invvar + stats_invvar_[j];
      const double post_mean = (stats_error_[j] + scaled_prior_mean) / invvar;
      marginal_lik += (post_mean * post_mean * invvar) - log(invvar);
    }
    r[i] = marginal_lik;
  }
}

void GaussianPriorIntegratedOptimize::UpdateVariance(
    ArraySlice<double> stats_invvar,
    ArraySlice<double> stats_error,
    ArraySlice<double> coefficients,
    util::random::Distribution* rng) {
  vector<double> inv_vars, likelihoods;
  int max_index = 0;
  const int num_levels = coefficients.size();

  ArraySlice<double> error(stats_error, 0, num_levels);
  ArraySlice<double> invvar(stats_invvar, 0, num_levels);
  GaussianIntegratedLikelihoodForVarianceUpdate cback(invvar, error);
  cback.SetPriorMean(this->GetPriorMean());
  NonNegativeOptimize(cback, this->GetPriorInverseVariance(),
                      this->GetPriorOptimConfig().grid_size,
                      this->GetPriorOptimConfig().num_iterations,
                      &inv_vars, &likelihoods, &max_index);
  this->SetPriorInverseVariance(inv_vars[max_index]);
}

namespace {
class NormalPriorLoglik : public LogLikelihodFunction<double> {
 public:
  virtual ~NormalPriorLoglik() {}

  explicit NormalPriorLoglik(ArraySlice<double> coefficients) {
    sum_coefficients_squared_ = std::accumulate(coefficients.begin(),
                                                coefficients.end(), 0.0,
        [](double accum, double s) { return accum + s * s; });
    num_coefficients_ = static_cast<double>(coefficients.size());
  }

  // Assumes that 'r' is non-NULL and has length at least that of 'x'
  void Evaluate(ArraySlice<double> x,
                MutableArraySlice<double> r) const override {
    for (int i = 0; i < x.size(); ++i) {
      const double inverse_variance = exp(x[i]);
      r[i] = -0.5 * sum_coefficients_squared_ * inverse_variance
             + 0.5 * x[i] * num_coefficients_;
    }
  }

 private:
  double sum_coefficients_squared_;
  double num_coefficients_;
};

}  // namespace

void GaussianPriorGibbsSampler::UpdateProposalStdDev() {
  if (num_steps_ < 20) {
    return;
  } else if (num_accepted_ < 0.2 * num_steps_) {
    num_steps_ = 0;
    num_accepted_ = 0;
    proposal_sd_ *= 0.5;
    LOG(INFO) << "reduced proposal_sd to " << proposal_sd_;
  } else if (num_accepted_ > 0.8 * num_steps_) {
    num_steps_ = 0;
    num_accepted_ = 0;
    proposal_sd_ *= 2.0;
    LOG(INFO) << "increased proposal_sd to " << proposal_sd_;
  }
}

void GaussianPriorGibbsSampler::UpdateVariance(
    ArraySlice<double> stats_invvar,
    ArraySlice<double> stats_error,
    ArraySlice<double> coefficients,
    util::random::Distribution* rng) {
  metropolis_hastings::SymmetricGaussianMhProposer proposer(proposal_sd_, rng);
  NormalPriorLoglik llik_cback(coefficients);

  const int num_steps = this->GetPriorOptimConfig().num_iterations;
  int num_accepted = 0;
  const double new_log_invvar = metropolis_hastings::RunMetropolisHastings(
      llik_cback, log(this->GetPriorInverseVariance()), num_steps, &proposer,
      rng, &num_accepted);
  num_accepted_ += num_accepted;
  num_steps_ += num_steps;
  this->SetPriorInverseVariance(exp(new_log_invvar));
  this->UpdateProposalStdDev();
}

}  // namespace emre
