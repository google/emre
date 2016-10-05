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

#include "likelihoods.h"  // NOLINT

#include <math.h>
#include "gsl/gsl_sf_gamma.h"

namespace emre {
namespace likelihoods {

double GammaNormLogConst(double alpha_param, double beta_param) {
  return alpha_param * log(beta_param) - gsl_sf_lngamma(alpha_param);
}

double GammaLogLik(double x, double alpha_param, double beta_param) {
  if (alpha_param <= 0.0) {
    return 0.0;
  }
  const double t1 = GammaNormLogConst(alpha_param, beta_param);
  const double t2 = log(x) * (alpha_param - 1.0);
  const double t3 = -beta_param * x;
  return t1 + t2 + t3;
}

double PoissonLogLik(double x, double lambda) {
  double t1 = 0.0;
  double t2 = 0.0;
  if (x > 0.0) {
    t1 = log(lambda) * x;
    t2 = -gsl_sf_lngamma(x + 1.0);
  }
  return -lambda + t1 + t2;
}

double UnnormalizedPoissonLogLik(double x, double lambda) {
  double t1 = 0.0;
  if (x > 0.0) {
    t1 = log(lambda) * x;
  }
  return -lambda + t1;
}

double PoissonGammaMargLogLik(
    double poisson_obs, double alpha_param, double beta_param,
    double tau_param) {
  const double t1 = GammaNormLogConst(alpha_param, beta_param);
  const double t2 = GammaNormLogConst(alpha_param + poisson_obs,
                                      beta_param + tau_param);
  const double t3 = poisson_obs * log(tau_param);
  return t1 - t2 + t3 - gsl_sf_lngamma(poisson_obs + 1.0);
}

}  // namespace likelihoods
}  // namespace emre
