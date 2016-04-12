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

// These likelihoods are used in prior and random effect updates.

#ifndef EMRE_PRIOR_UPDATER_LIKELIHOODS_H_  // NOLINT
#define EMRE_PRIOR_UPDATER_LIKELIHOODS_H_

namespace emre {
namespace likelihoods {

// These functions are used to compute the marginal likelihood of Poisson
// observations under a Gamma-Poisson model.
double GammaNormLogConst(double alpha_param, double beta_param);

double GammaLogLik(double x, double alpha_param, double beta_param);

double PoissonLogLik(double x, double lambda);
double UnnormalizedPoissonLogLik(double x, double lambda);

double PoissonGammaMargLogLik(double poisson_obs, double alpha_param,
                              double beta_param, double tau_param);

}  // namespace likelihoods
}  // namespace emre

#endif  // EMRE_PRIOR_UPDATER_LIKELIHOODS_H_  // NOLINT
