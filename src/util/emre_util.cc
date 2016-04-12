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

#include "emre_util.h"  // NOLINT

#include <assert.h>
#include <algorithm>

namespace emre {
namespace util {

// static
int EmreUtil::MultinomialFromPdf(ArraySlice<double> pdf, double p) {
  double sum = 0.0;
  int idx = 0;
  for (double x : pdf) {
    sum += x;
    if (p <= sum) {
      return idx;
    }
    ++idx;
  }
  return -1;
}

// static
void EmreUtil::LoglikelihoodsToProbs(ArraySlice<double> lliks,
                                      MutableArraySlice<double> r) {
  assert(lliks.size() == r.size());
  const double max_value = *std::max_element(lliks.begin(), lliks.end());
  double likelihood_sum = 0.0;
  for (int i = 0; i < lliks.size(); ++i) {
    r[i] = exp(lliks[i] - max_value);
    likelihood_sum += r[i];
  }
  // normalize
  for (int i = 0; i < lliks.size(); ++i) {
    r[i] /= likelihood_sum;
  }
}

// static
void EmreUtil::LoglikelihoodsToLogProbs(ArraySlice<double> lliks,
                                         MutableArraySlice<double> r) {
  assert(lliks.size() == r.size());
  const double max_value = *std::max_element(lliks.begin(), lliks.end());
  double likelihood_sum = 0.0;
  for (int i = 0; i < lliks.size(); ++i) {
    likelihood_sum += exp(lliks[i] - max_value);
  }
  const double log_sum = log(likelihood_sum);
  for (int i = 0; i < lliks.size(); ++i) {
    r[i] = lliks[i] - max_value - log_sum;
  }
}

void EmreUtil::GetVectorSubset(
    ArraySlice<double> v,
    MutableArraySlice<double> r) {
  const int r_size = r.size();
  const int v_size = v.size();

  int step_size = 0;
  for (int x : {1097, 1103, 1109, 1117, 1123}) {
    if (v_size % x != 0) {
      step_size = x;
      break;
    }
  }
  assert(step_size != 0);
  //  CHECK(step_size != 0) << v_size << " is divisible by all "
  //      << "five of the following primes: 1097, 1103, 1109, 1117, 1123.";
  for (int i = 0, j = 0; j < r_size; ++j) {
    r[j] = v[i];
    i = (i + step_size) % v_size;
  }
}

}  // namespace util
}  // namespace emre
