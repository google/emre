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

#include <algorithm>
#include <vector>

#include "gaussian_prior_optimize.h"  // NOLINT
#include "testing/base/public/gunit.h"

namespace emre {

using util::ArraySlice;
using util::MutableArraySlice;

TEST(GaussianPriorOptimizeTest, DoesSampleOptimize) {
  // Generated from the R code:
  //   set.seed(15); v <- rnorm(100, 0, 3); paste(round(v, 3), collapse = ", ")
  const double kInverseVariance = 1.0 / (3.0 * 3.0);
  const double kDataStdDev = 2.981873;
  const vector<double> sample_gaussian = {
      0.776, 5.493, -1.019, 2.692, 1.464, -3.766, 0.068, 3.272, -0.396, -3.225,
      2.565, -1.095, 0.497, -3.728, 4.378, -0.011, -0.063, 0.096, -3.502,
      -1.559, 4.122, 4.237, -1.207, -1.317, 3.032, 1.292, 2.202, -2.042, 0.979,
      2.721, -1.388, 0.014, 4.351, 2.258, 2.898, 1.405, -0.723, 3.098, -1.921,
      -3.949, 1.086, 2.164, 7.456, -2.955, -1.015, 3.763, -3.385, 4.679, 2.124,
      2.803, 2.051, 3.058, -0.721, 5.033, 1.228, -2.666, -2.097, 1.56, 3.471,
      -0.332, 0.896, -1.283, -1.768, -3.89, -4.463, -3.481, -0.998, 1.627,
      -2.221, 3.724, -4.576, -5.631, -3.981, -1.491, 0.12, -1.634, 1.08, 3.009,
      -2.946, 6.121, 0.01, 7.005, 0.463, -1.397, -1.409, -1.49, -0.967, 1.844,
      0.383, -0.455, -2.328, -3.954, 6.312, 1.137, -0.672, 4.656, -0.51, 3.854,
      -3.006, -7.312
  };
  const vector<double> stats_invvar(sample_gaussian.size(), 1.0);

  GaussianPriorSampleOptimize optimizer;

  FeatureFamilyPrior prior;
  prior.set_mean(0.0);
  prior.set_inverse_variance(1.0);

  optimizer.SetPriorFromProto(prior);
  optimizer.UpdateVariance(stats_invvar, sample_gaussian,
                           sample_gaussian, nullptr /* rng */);
  optimizer.SetProtoFromPrior(&prior);

  CHECK_EQ(prior.mean(), 0.0);

  // Check that estimate is near truth
  CHECK_NEAR(prior.inverse_variance(), kInverseVariance, 0.05);

  // Check for regressions
  CHECK_NEAR(prior.inverse_variance(),
             1.0 / (kDataStdDev * kDataStdDev),
             0.001);
}

TEST(GaussianPriorOptimizeTest, DoesIntegratedOptimize) {
  // Generated from the R code:
  //   set.seed(15)
  //   re <- rnorm(100, 0, 3)
  //   noise.sd <- rexp(length(re))^-0.5
  //   obs <- re + noise.sd * rnorm(length(re))
  //   paste(round(re, 3), collapse = ", ")
  //   paste(round(obs, 3), collapse = ", ")
  //   paste(round(noise.sd, 3), collapse = ", ")

  static const double kInverseVariance = 1.0 / (3.0 * 3.0);

  const vector<double> sample_ranefs = {
      0.776, 5.493, -1.019, 2.692, 1.464, -3.766, 0.068, 3.272, -0.396, -3.225,
      2.565, -1.095, 0.497, -3.728, 4.378, -0.011, -0.063, 0.096, -3.502,
      -1.559, 4.122, 4.237, -1.207, -1.317, 3.032, 1.292, 2.202, -2.042, 0.979,
      2.721, -1.388, 0.014, 4.351, 2.258, 2.898, 1.405, -0.723, 3.098, -1.921,
      -3.949, 1.086, 2.164, 7.456, -2.955, -1.015, 3.763, -3.385, 4.679, 2.124,
      2.803, 2.051, 3.058, -0.721, 5.033, 1.228, -2.666, -2.097, 1.56, 3.471,
      -0.332, 0.896, -1.283, -1.768, -3.89, -4.463, -3.481, -0.998, 1.627,
      -2.221, 3.724, -4.576, -5.631, -3.981, -1.491, 0.12, -1.634, 1.08, 3.009,
      -2.946, 6.121, 0.01, 7.005, 0.463, -1.397, -1.409, -1.49, -0.967, 1.844,
      0.383, -0.455, -2.328, -3.954, 6.312, 1.137, -0.672, 4.656, -0.51, 3.854,
      -3.006, -7.312};

  const vector<double> sample_obs = {
      1.482, 4.781, -2.652, 3.922, 2.014, -4.529, -9.682, 3.841, -0.369, -0.946,
      1.634, -2.604, -0.904, -2.47, -0.49, 2.606, 1.101, -0.568, -2.294, -1.652,
      4.483, 4.215, -1.058, -2.586, -0.107, 0.752, 2.415, -0.371, 3.158, 3.278,
      -1.148, -0.91, 5.691, 4.106, 3.559, 2.749, -4.433, 3.313, -1.586, -4.119,
      0.745, 3.21, 5.673, -12.718, -2.209, 3.827, -3.688, 6.284, 2.681, 1.846,
      1.302, 1.76, -1.007, 4.463, 4.584, -2.904, -1.118, -3.024, 3.433, -12.748,
      1.053, -1.794, -2.187, -3.15, -5.593, -3.684, 0.386, 5.66, -1.498, 5.035,
      -2.839, -5.603, 0.135, 0.905, 1.284, -0.791, 1.48, -5.433, -0.444, 4.412,
      -0.27, 6.738, -0.198, 2.094, -2.211, -1.483, 0.418, 2.507, 0.784, -1.404,
      -2.924, -2.473, -19.617, 0.058, -0.218, 2.596, -1.644, 4.023, -3.249,
      -10.502};

  const vector<double> obs_sd = {
      0.985, 0.622, 1.618, 0.925, 0.955, 1.247, 14.509, 0.443, 1.268, 1.761,
      0.624, 0.722, 1.063, 1.463, 15.967, 3.009, 1.638, 0.91, 1.475, 1.529,
      0.685, 5.169, 0.649, 1.147, 1.733, 0.872, 0.765, 1.735, 1.099, 0.682,
      0.641, 1.217, 0.749, 1.529, 1.032, 1.014, 5.286, 0.434, 1.173, 0.712,
      0.9, 2.281, 3.207, 2.319, 2.28, 1.935, 1.191, 1.123, 0.561, 1.282,
      0.985, 1.466, 0.489, 1.195, 4.617, 0.706, 1.257, 2.51, 1.458, 7.714,
      1.583, 1.706, 0.858, 1.186, 1.054, 1.525, 1.451, 5.757, 0.526, 2.109,
      1.19, 1.192, 2.527, 2.023, 1, 0.664, 0.634, 6.544, 4.338, 1.144, 1.021,
      0.471, 0.879, 3.364, 0.949, 1.054, 0.983, 1.751, 0.675, 0.901, 0.999,
      1.99, 12.293, 0.694, 0.923, 1.586, 1.271, 0.887, 1.5, 2.52};

  const int kNumRanefs = sample_ranefs.size();
  {
    vector<double> stats_invvar(kNumRanefs);
    vector<double> stats_error(kNumRanefs);
    for (int i = 0; i < kNumRanefs; ++i) {
      const double invvar = 1.0 / (obs_sd[i] * obs_sd[i]);
      stats_error[i] = invvar * sample_obs[i];
      stats_invvar[i] = invvar;
    }

    GaussianPriorIntegratedOptimize optimizer;

    FeatureFamilyPrior prior;
    prior.set_inverse_variance(1.0);

    optimizer.SetPriorFromProto(prior);
    optimizer.UpdateVariance(stats_invvar, stats_error,
                             sample_ranefs, nullptr /* rng */);
    optimizer.SetProtoFromPrior(&prior);

    CHECK_EQ(prior.mean(), 0.0);

    // Check that estimate is near truth
    CHECK_NEAR(prior.inverse_variance(), kInverseVariance, 0.02);

    // Check for regressions
    CHECK_NEAR(prior.inverse_variance(), 0.121717, 0.001);
  }

  {
    static const double kMeanShift = 2.0;
    vector<double> stats_invvar(kNumRanefs);
    vector<double> stats_error(kNumRanefs);
    for (int i = 0; i < kNumRanefs; ++i) {
      const double invvar = 1.0 / (obs_sd[i] * obs_sd[i]);
      stats_error[i] = invvar * (sample_obs[i] + kMeanShift);
      stats_invvar[i] = invvar;
    }

    GaussianPriorIntegratedOptimize optimizer;

    FeatureFamilyPrior prior;
    prior.set_mean(kMeanShift);
    prior.set_inverse_variance(1.0);

    optimizer.SetPriorFromProto(prior);
    optimizer.UpdateVariance(stats_invvar, stats_error,
                             sample_ranefs, nullptr /* rng */);
    optimizer.SetProtoFromPrior(&prior);

    CHECK_EQ(prior.mean(), kMeanShift);

    // Check that estimate is near truth
    CHECK_NEAR(prior.inverse_variance(), kInverseVariance, 0.02);

    // Check for regressions
    CHECK_NEAR(prior.inverse_variance(), 0.121717, 0.001);
  }
}

TEST(GaussianPriorOptimizeTest, DoesGibbsSamplePrior) {
  // Generated from the R code:
  //   set.seed(15)
  //   re <- rnorm(100, 0, 3)
  //   paste(round(re, 3), collapse = ", ")

  static const double kInverseVariance = 1.0 / (3.0 * 3.0);

  const vector<double> sample_ranefs = {
      0.776, 5.493, -1.019, 2.692, 1.464, -3.766, 0.068, 3.272, -0.396, -3.225,
      2.565, -1.095, 0.497, -3.728, 4.378, -0.011, -0.063, 0.096, -3.502,
      -1.559, 4.122, 4.237, -1.207, -1.317, 3.032, 1.292, 2.202, -2.042, 0.979,
      2.721, -1.388, 0.014, 4.351, 2.258, 2.898, 1.405, -0.723, 3.098, -1.921,
      -3.949, 1.086, 2.164, 7.456, -2.955, -1.015, 3.763, -3.385, 4.679, 2.124,
      2.803, 2.051, 3.058, -0.721, 5.033, 1.228, -2.666, -2.097, 1.56, 3.471,
      -0.332, 0.896, -1.283, -1.768, -3.89, -4.463, -3.481, -0.998, 1.627,
      -2.221, 3.724, -4.576, -5.631, -3.981, -1.491, 0.12, -1.634, 1.08, 3.009,
      -2.946, 6.121, 0.01, 7.005, 0.463, -1.397, -1.409, -1.49, -0.967, 1.844,
      0.383, -0.455, -2.328, -3.954, 6.312, 1.137, -0.672, 4.656, -0.51, 3.854,
      -3.006, -7.312};

  const int kNumRanefs = sample_ranefs.size();
  vector<double> stats_invvar(kNumRanefs);
  vector<double> stats_error(kNumRanefs);

  util::random::Distribution distn(15);
  GaussianPriorGibbsSampler sampler;

  FeatureFamilyPrior prior;
  prior.set_inverse_variance(1.0);
  sampler.SetPriorFromProto(prior);
  sampler.SetProposalStdDev(10.0);

  const int num_burnin = 40;
  const int num_steps = 1000;

  for (int i = 0; i < num_burnin; ++i) {
    sampler.UpdateVariance(stats_invvar, stats_error,
                           sample_ranefs, &distn);
    sampler.SetProtoFromPrior(&prior);
  }

  const double proposal_sd = sampler.GetProposalStdDev();
  EXPECT_NEAR(proposal_sd, 0.5, 0.3);

  vector<double> log_invvars;
  for (int i = 0; i < num_steps; ++i) {
    sampler.UpdateVariance(stats_invvar, stats_error,
                           sample_ranefs, &distn);
    FeatureFamilyPrior prior;
    sampler.SetProtoFromPrior(&prior);
    CHECK_EQ(prior.mean(), 0.0);
    CHECK_GT(prior.inverse_variance(), 0.0);
    log_invvars.push_back(log(prior.inverse_variance()));
  }

  const double scl = 1.0 / num_steps;
  const double log_mean = std::accumulate(
      log_invvars.begin(), log_invvars.end(), 0.0) * scl;
  const double log_square_sum = std::accumulate(
      log_invvars.begin(), log_invvars.end(), 0.0,
      [] (double x, double y) -> double { return x + y * y; });
  const double log_var = log_square_sum * scl - log_mean * log_mean;
  const double log_sd = sqrt(log_var);
  EXPECT_NEAR(log_sd, 0.12568, 0.02);
  EXPECT_NEAR(log_mean, log(kInverseVariance), 0.03);
}


TEST(PriorOptimizeTest, DoesNothingPriorUpdater) {
  typedef FeatureFamilyPrior FFP;
  vector<double> stats_invvar;
  vector<double> sample_gaussian;

  FeatureFamilyPrior init_prior;
  init_prior.set_feature_family("w.1");
  init_prior.set_mean(1.0);
  init_prior.set_inverse_variance(17.0);
  init_prior.set_model_class_type(FFP::GAUSSIAN);
  init_prior.set_ranef_update_type(FFP::OPTIMIZED);
  init_prior.set_prior_update_type(FFP::GIBBS_INTEGRATED);

  DoesNothingPriorUpdater prior_optim;
  prior_optim.SetPriorFromProto(init_prior);
  prior_optim.UpdateVariance(stats_invvar, sample_gaussian,
                            sample_gaussian, nullptr /* rng */);

  FeatureFamilyPrior out_prior;
  prior_optim.SetProtoFromPrior(&out_prior);
  EXPECT_EQ(out_prior.feature_family(), init_prior.feature_family());
  EXPECT_EQ(out_prior.mean(), init_prior.mean());
  EXPECT_EQ(out_prior.inverse_variance(), init_prior.inverse_variance());
  EXPECT_EQ(out_prior.model_class_type(), init_prior.model_class_type());
  EXPECT_EQ(out_prior.prior_update_type(), init_prior.prior_update_type());
  EXPECT_EQ(out_prior.ranef_update_type(), init_prior.ranef_update_type());
}

}  // namespace emre
