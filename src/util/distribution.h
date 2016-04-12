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

#ifndef _EMRE_UTIL_DISTRIBUTION_H_  // NOLINT
#define _EMRE_UTIL_DISTRIBUTION_H_

#include <memory>

#include "third_party/gsl/gsl/gsl_rng.h"

namespace emre {
namespace util {
namespace random {

class Distribution {
 public:
  explicit Distribution(int seed);
  ~Distribution() {
    if (rng_ != nullptr) {
      gsl_rng_free(rng_);
      rng_ = nullptr;
    }
  }
  // Returns true with probability p, false with (1 - p).
  // Requires: 0.0 <= p <= 1.0.
  bool RandBernoulli(double p);

  // Return a random number drawn from the Beta distribution Beta(a, b).
  // Requires: a > 0.0, b > 0.0.
  double RandBeta(double a, double b);

  // Requires: k, theta > 0.0.
  double RandGamma(double k, double theta);

  // Return a random number drawn from a Gaussian (normal) distribution with
  // mean zero and standard deviation sigma.
  // Requires: sigma > 0
  double RandGaussian(double sigma);

 private:
  gsl_rng* rng_;
};

}  // namespace random
}  // namespace util
}  // namespace emre

#endif  // _EMRE_UTIL_DISTRIBUTION_H_  // NOLINT
