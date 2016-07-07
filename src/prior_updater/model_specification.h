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

// This library abstracts the likelihoods need to fit three types of priors:
//   a) grid prior: Pr(A) = \sum_k w_k 1{z_k \in A}
//   b) mixture prior:  Pr(A) = \sum_k w_k F(A; theta_k)
//        e.g. mixture of Gammas or mixture of Gaussians
//   c) spike prior: Pr(A) = w * 1{spike \in A} + (1 - w) * F(A; theta)
//        e.g. spike = 0 and F() = N(0, theta^2)
//        e.g. spike = 1 and F() = Gamma(theta, theta)

#ifndef EMRE_PRIOR_UPDATER_MODEL_SPECIFICATION_H_  // NOLINT
#define EMRE_PRIOR_UPDATER_MODEL_SPECIFICATION_H_

#include "ranef_updater/block_relaxation.h"
#include "util/distribution.h"

namespace emre {

class ModelSpecification {
 public:
  virtual ~ModelSpecification() {}

  // The marginal likelihood of data should be a sum of
  //    exp(CalculatePriorLLik() + CalculatePosteriorLLik())
  // where the sum is over mixture components.  These two functions are
  // broken out for efficiency: one of them does not depend on the data
  // and its value can be re-used.
  virtual double CalculatePriorLLik(const PriorComponent& x) = 0;
  virtual double CalculatePosteriorLLik(const PriorComponent& x,
                                        double stats_first,
                                        double stats_second) = 0;

  virtual BlockRelaxation::MeanVar CalculatePosteriorMeanVariance(
      const PriorComponent& x,
      double stats_mean, double stats_invvar) {
    return BlockRelaxation::MeanVar{0.0, 0.0};
  }

  // Takes a non-negative value of 'param' and updates the appropriate fields
  // in 'r'.  For example, param can specifify the standard deviation in a
  // Gaussian prior with mean of 0.0.  Another example is a scale parameter in
  // a Gamma prior.
  virtual void SetMixtureParameter(double param, PriorComponent* r) = 0;

  // Extracts the parameter that SetMixtureParameter above sets in the proto
  virtual double GetMixtureParameter(const PriorComponent& x) = 0;

  // This is used for Gibbs sampling from the mixture distribution.
  virtual double SampleFromPosterior(const PriorComponent& x,
                                     double stats_first, double stats_second,
                                     util::random::Distribution* rng) = 0;

  // This is specific to the spike prior e.g. in Gamma-Poisson it would be 1.0
  // and in a Gaussian-Gaussian model it would be 0.0
  virtual double SpikeLocation() const = 0;
  virtual double CalculateSpikeDataLoglik(double stats_first,
                                          double stats_second) = 0;

  // Used in the grid and spike prior e.g. in the Gamma-Poisson each grid point
  // is a location in (0, Inf) and in the Gaussian-Gaussian model it would live
  // in (-Inf, Inf)
  virtual double CalculateDataLoglik(double location,
                                     double stats_first,
                                     double stats_second) = 0;
};

}  // namespace emre

#endif  // EMRE_PRIOR_UPDATER_MODEL_SPECIFICATION_H_  // NOLINT
