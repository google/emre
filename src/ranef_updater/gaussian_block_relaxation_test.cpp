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

#include "gaussian_block_relaxation.h"  // NOLINT
#include "testing/base/public/gunit.h"

namespace emre {
namespace gaussian {

TEST(GaussianBlockRelaxationTest, DoesUpdate) {
  util::random::Distribution distn(15);

  FeatureFamilyPrior prior_pb;
  prior_pb.set_mean(0.0);
  prior_pb.set_inverse_variance(1.0 / (2.0 * 2.0));

  const vector<double> predicted = {0.0 * 0.01, -3.0 * 100.0};
  const vector<double> inv_var = {0.01, 100.0};
  {
    // The posterior mean for the samples should be 0.0 and -3.0 respectively.
    // The noise standard deviations are 0.01^-0.5 = 10 and 100^-0.5 = 0.1
    // so the first random effect should have a posterior std dev nearly equal
    // to the prior standard deviation of 2.0 and the second should be a
    // posterior standard deviation of just above 0.1
    GaussianGibbsSampler updater;
    updater.SetRng(&distn);

    vector<double> sample_scores(predicted.size());
    vector<double> sample_sum(predicted.size(), 0.0);
    vector<double> sample_sum_squared(predicted.size(), 0.0);

    const int kNumIterations = 1000;
    for (int i = 0; i < kNumIterations; ++i) {
      BlockRelaxation::UpdateParameters param(predicted, inv_var);
      updater.UpdateRanefs(prior_pb, param, &sample_scores);
      for (int j = 0; j < predicted.size(); ++j) {
        sample_sum[j] += sample_scores[j];
        sample_sum_squared[j] += sample_scores[j] * sample_scores[j];
      }
    }

    const vector<double> expected_var =
        {1.0 / (0.25 + 0.01), 1.0 / (0.25 + 100.0)};
    const vector<double> expected_var_tol = {0.2, 0.005};
    const vector<double> expected_mean = {0.0, -3.0 * 100.0 / (0.25 + 100.0)};
    const vector<double> expected_mean_tol = {0.1, 0.01};

    for (int j = 0; j < predicted.size(); ++j) {
      auto post = CalculatePosteriorMeanVariance(
          prior_pb.mean(), prior_pb.inverse_variance(),
          predicted[j], inv_var[j]);
      EXPECT_NEAR(post.mean, expected_mean[j], 1e-6);
      EXPECT_NEAR(post.var, expected_var[j], 1e-6);

      const double sample_mean = sample_sum[j] / kNumIterations;
      const double sample_var = (sample_sum_squared[j] / kNumIterations)
                                - sample_mean * sample_mean;
      EXPECT_NEAR(sample_mean, expected_mean[j], expected_mean_tol[j]);
      EXPECT_NEAR(sample_var, expected_var[j], expected_var_tol[j]);
    }
  }

  {
    GaussianOptimizer updater;
    vector<double> scores(predicted.size());
    BlockRelaxation::UpdateParameters param(predicted, inv_var);
    updater.UpdateRanefs(prior_pb, param, &scores);

    const vector<double> expected_scores = {0.0, -3.0 * 100.0 / (0.25 + 100.0)};
    for (int j = 0; j < predicted.size(); ++j) {
      EXPECT_NEAR(scores[j], expected_scores[j], 0.001);
    }
  }
}

}  // namespace gaussian
}  // namespace emre
