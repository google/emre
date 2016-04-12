// Copyright 2010-2016 Google
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

#include <vector>

#include "testing/base/public/gunit.h"

namespace emre {
namespace util {

class EmreUtilTest : public ::testing::Test {};

TEST(EmreUtilTest, MultinomialFromPdfTest) {
  const vector<double> pdf = {0.1, 0.7, 0.2};
  EXPECT_EQ(EmreUtil::MultinomialFromPdf(pdf, 0.0), 0);
  EXPECT_EQ(EmreUtil::MultinomialFromPdf(pdf, 0.1), 0);
  EXPECT_EQ(EmreUtil::MultinomialFromPdf(pdf, 0.15), 1);
  EXPECT_EQ(EmreUtil::MultinomialFromPdf(pdf, 0.79), 1);
  EXPECT_EQ(EmreUtil::MultinomialFromPdf(pdf, 0.81), 2);
  EXPECT_EQ(EmreUtil::MultinomialFromPdf(pdf, 1.0), 2);
}

TEST(RegmhUtilTest, LoglikelihoodsToProbsTest) {
  const vector<double> lliks = {log(0.1)+0.5, log(0.7)+0.5, log(0.2)+0.5};
  vector<double> probs(3, 0.0);

  EmreUtil::LoglikelihoodsToProbs(lliks, &probs);
  EXPECT_NEAR(probs[0], 0.1, 0.0001);
  EXPECT_NEAR(probs[1], 0.7, 0.0001);
  EXPECT_NEAR(probs[2], 0.2, 0.0001);
}

TEST(EmreUtilTest, LoglikelihoodsToLogProbs) {
  const vector<double> lliks = {log(0.1)+0.5, log(0.7)+0.5, log(0.2)+0.5};
  vector<double> log_probs(3, 0.0);

  EmreUtil::LoglikelihoodsToLogProbs(lliks, &log_probs);
  EXPECT_NEAR(log_probs[0], log(0.1), 0.0001);
  EXPECT_NEAR(log_probs[1], log(0.7), 0.0001);
  EXPECT_NEAR(log_probs[2], log(0.2), 0.0001);
}


TEST(EmreUtilTest, GetVectorSubset) {
  const vector<double> v = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
  vector<double> v_subset(3, 0.0);

  EmreUtil::GetVectorSubset(v, &v_subset);
  const int idx1 = 1097 % 6;
  const int idx2 = (idx1 + 1097) % 6;
  EXPECT_EQ(v_subset[0], v[0]);
  EXPECT_EQ(v_subset[1], v[idx1]);
  EXPECT_EQ(v_subset[2], v[idx2]);
}

}  // namespace util
}  // namespace emre
