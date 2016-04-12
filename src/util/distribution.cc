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

#include "distribution.h"  // NOLINT

#include "gsl/gsl_randist.h"

namespace emre {
namespace util {
namespace random {

Distribution::Distribution(int seed) {
  // initialize with Mersenne twister
  gsl_rng_default_seed = seed;
  rng_ = gsl_rng_alloc(gsl_rng_mt19937);
}

double Distribution::RandBeta(double a, double b) {
  return gsl_ran_beta(rng_, a, b);
}

bool Distribution::RandBernoulli(double p) {
  unsigned int v = gsl_ran_bernoulli(rng_, p);
  return v == 1;
}

double Distribution::RandGamma(double k, double theta) {
  return gsl_ran_gamma(rng_, k, theta);
}

double Distribution::RandGaussian(double sigma) {
  return gsl_ran_gaussian(rng_, sigma);
}

}  // namespace random
}  // namespace util
}  // namespace emre
