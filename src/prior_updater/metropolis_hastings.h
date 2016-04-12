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

#ifndef EMRE_PRIOR_UPDATER_METROPOLIS_HASTINGS_H_  // NOLINT
#define EMRE_PRIOR_UPDATER_METROPOLIS_HASTINGS_H_

#include <math.h>
#include <map>
#include <vector>

#include "likelihoods.h"  // NOLINT
#include "contentads/analysis/caa/search_plus/regmh/emre/src/util/arrayslice.h"
#include "contentads/analysis/caa/search_plus/regmh/emre/src/util/basic_types.h"
#include "contentads/analysis/caa/search_plus/regmh/emre/src/util/distribution.h"

namespace emre {
namespace metropolis_hastings {

using util::ArraySlice;

// Metropolis-Hastings proposal.  This is used to generate the proposals in an
// MCMC update of model prior parameters.
template<typename T> class MhProposer {
 public:
  virtual ~MhProposer() {}

  virtual void GenerateProposal(const T& from, T* to,
                                double* proposal_llik) = 0;

  // only used to compute the Metropolis-Hastings ratio.  Can be 0.0 for
  // symmetric proposal distributions
  virtual double ProposalLoglikForMhRatio(const T& from, const T& to) = 0;
};

class SymmetricGaussianMhProposer : public MhProposer<double> {
 public:
  // does not own 'distn'
  SymmetricGaussianMhProposer(double proposal_sd,
                              util::random::Distribution* distn)
      : proposal_sd_(proposal_sd), distn_(distn) {}

  virtual ~SymmetricGaussianMhProposer() {}

  void GenerateProposal(const double& from, double* to,
                        double* proposal_llik) override {
    if (proposal_llik) *proposal_llik = 0.0;
    *to = from + distn_->RandGaussian(proposal_sd_);
  }

  double ProposalLoglikForMhRatio(const double& from,
                                  const double& to) override {
    return 0.0;
  }

 private:
  const double proposal_sd_;
  util::random::Distribution* distn_;
};


class LangevinMCMCProposer : public MhProposer<vector<double>> {
 public:
  // does not own 'distn'
  // 'prior' is used only for the locations of the grid points
  LangevinMCMCProposer(int dimension,
                       double proposal_sd,
                       util::random::Distribution* distn);

  virtual ~LangevinMCMCProposer() {}

  void GenerateProposal(const vector<double>& from,
                        vector<double>* to,
                        double* proposal_llik) override;

  double ProposalLoglikForMhRatio(const vector<double>& from,
                                  const vector<double>& to) override;

 protected:
  int GetDimension() const { return dimension_; }
  util::random::Distribution* GetRng() { return distn_; }
  virtual void GetLlikGradient(ArraySlice<double> from,
                               vector<double>* gr) = 0;

 private:
  void GetProposalMean(const vector<double>& from,
                       vector<double>* proposal_mean);


 private:
  const int dimension_;
  const double proposal_sd_;
  const double proposal_invvar_;  // proposal_sd_^-2
  const double gradient_stepsize_;  // 0.5 * proposal_sd_^2
  util::random::Distribution* distn_;

  // This pre-allocated vector is used to avoid repeated allocations during
  // calls to ProposalLoglikForMhRatio
  vector<double> temp_vector_;
};


template<typename T>
static T RunMetropolisHastings(
    const LogLikelihodFunction<T>& llik_cback,
    T initial_value, int num_steps,
    MhProposer<T>* proposal_generator,
    util::random::Distribution* distn,
    int* num_accepted) {
  int num_rejected = 0;

  // The current and proposal points are wrapped in vectors because
  // a) llik_cback.Evaluate() takes a vector as its argument
  // b) we use vector's swap function to swap the values without constructing
  //    a copy
  vector<T> x = {initial_value};
  vector<T> x_star = {initial_value};
  vector<double> llik = {0.0};
  llik_cback.Evaluate(x, &llik);
  double x_llik = llik[0];

  for (int i = 0; i < num_steps; ++i) {
    double fwd_llik;
    proposal_generator->GenerateProposal(x[0], &x_star[0], &fwd_llik);
    llik[0] = 0.0;
    llik_cback.Evaluate(x_star, &llik);
    const double x_star_llik = llik[0];
    const double rev_llik =
        proposal_generator->ProposalLoglikForMhRatio(x_star[0], x[0]);
    const double log_mh_ratio = x_star_llik + rev_llik - x_llik - fwd_llik;

    if (log_mh_ratio > 0.0 || distn->RandBernoulli(exp(log_mh_ratio))) {
      x.swap(x_star);
      x_llik = x_star_llik;
    } else {
      ++num_rejected;
    }
  }
  if (num_accepted != nullptr) {
    *num_accepted = num_steps - num_rejected;
  }
  return x[0];
}

}  // namespace metropolis_hastings
}  // namespace emre

#endif  // EMRE_PRIOR_UPDATER_METROPOLIS_HASTINGS_H_  // NOLINT

