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

#ifndef EMRE_UTIL_EMRE_UTIL_H_  // NOLINT
#define EMRE_UTIL_EMRE_UTIL_H_

#include "arrayslice.h"  // NOLINT

namespace emre {
namespace util {

class EmreUtil {
 public:
  // If the second argument p ~ Unif[0, 1], then the output will have
  // multinomial distribution with probabilities in 'pdf'
  static int MultinomialFromPdf(ArraySlice<double> pdf, double p);

  // Exponentiates and renormalizes the log-likelihoods in lliks to get
  // probabilities.
  static void LoglikelihoodsToProbs(ArraySlice<double> lliks,
                                    MutableArraySlice<double> r);

  // A more numerically accurate version of the function
  // LoglikelihoodsToProbs which returns values in log scale
  static void LoglikelihoodsToLogProbs(ArraySlice<double> lliks,
                                       MutableArraySlice<double> r);

  static void GetVectorSubset(ArraySlice<double> v,
                              MutableArraySlice<double> r);
};

}  // namespace util
}  // namespace emre

#endif  // EMRE_UTIL_EMRE_UTIL_H_  // NOLINT
