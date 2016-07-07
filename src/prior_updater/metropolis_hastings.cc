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

#include "metropolis_hastings.h"  // NOLINT

#include "base/logging.h"

namespace emre {
namespace metropolis_hastings {

LangevinMCMCProposer::LangevinMCMCProposer(
    int dimension,
    double proposal_sd,
    util::random::Distribution* distn)
    : dimension_(dimension),
      proposal_sd_(proposal_sd),
      proposal_invvar_(1.0 / (proposal_sd_ * proposal_sd_)),
      gradient_stepsize_(0.5 * proposal_sd_ * proposal_sd_),
      distn_(distn), temp_vector_(dimension, 0.0) {}

void LangevinMCMCProposer::GenerateProposal(
    const vector<double>& from,
    vector<double>* to,
    double* proposal_llik) {
  CHECK_EQ(dimension_, from.size());
  CHECK_EQ(dimension_, to->size());
  this->GetProposalMean(from, to);

  double sum_squared = 0.0;  // holds sum of squared step sizes
  for (int i = 0; i < dimension_; ++i) {
    const double step_noise = distn_->RandGaussian(proposal_sd_);
    (*to)[i] += step_noise;
    sum_squared += step_noise * step_noise;
  }

  if (proposal_llik) {
    *proposal_llik = -0.5 * sum_squared * proposal_invvar_;
  }
}

double LangevinMCMCProposer::ProposalLoglikForMhRatio(
    const vector<double>& from,
    const vector<double>& to) {
  vector<double>* proposal_mean = &temp_vector_;
  this->GetProposalMean(from, proposal_mean);

  double sum_squared = 0.0;  // holds sum of squared step sizes
  for (int i = 0; i < dimension_; ++i) {
    const double step_noise = (*proposal_mean)[i] - to[i];
    sum_squared += step_noise * step_noise;
  }
  return -0.5 * sum_squared * proposal_invvar_;
}

void LangevinMCMCProposer::GetProposalMean(const vector<double>& from,
                                           vector<double>* proposal_mean) {
  // After this call 'proposal_mean' holds the gradient of the log likelihood
  this->GetLlikGradient(from, proposal_mean);

  const int num_components = this->GetDimension();
  for (int i = 0; i < num_components; ++i) {
    // to = gradient * proposal_sd + from
    (*proposal_mean)[i] = (*proposal_mean)[i] * gradient_stepsize_ + from[i];
  }
}

}  // namespace metropolis_hastings
}  // namespace emre
