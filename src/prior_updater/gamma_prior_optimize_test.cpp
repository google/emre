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

#include <vector>

#include "gamma_prior_optimize.h"  // NOLINT
#include "testing/base/public/gunit.h"

namespace emre {

using util::ArraySlice;
using util::MutableArraySlice;

TEST(GammaPriorOptimizeTest, DoesQuadraticOptimize) {
  // This test verifies that NonnegativeOptimize accurately maximizes a
  // quadratic
  const double kMaximumPoint = 3.0;

  class QuadraticFunc : public RealFunction {
   public:
    explicit QuadraticFunc(double x) : max_(x) {}

    void Evaluate(ArraySlice<double> x,
                  MutableArraySlice<double> r) const override {
      for (int i = 0; i < x.size(); ++i) {
        r[i] = -(x[i] - max_) * (x[i] - max_);
      }
    }
   private:
    double max_;
  };

  QuadraticFunc cback(kMaximumPoint);
  static const double kInitialValue = 1.0;

  {
    static const int kGridSize = 10;
    static const int kNumZoom = 5;
    vector<double> evaluation_points, values;
    int max_index = -1;
    NonNegativeOptimize(
        cback, kInitialValue, kGridSize, kNumZoom, &evaluation_points, &values,
        &max_index);
    EXPECT_NEAR(evaluation_points[max_index], kMaximumPoint, 1e-2);
  }

  {
    static const int kGridSize = 30;
    static const int kNumZoom = 5;
    vector<double> evaluation_points, values;
    int max_index = -1;
    NonNegativeOptimize(
        cback, kInitialValue, kGridSize, kNumZoom, &evaluation_points, &values,
        &max_index);
    EXPECT_NEAR(evaluation_points[max_index], kMaximumPoint, 1e-4);
  }
}


TEST(GammaPriorOptimizeTest, DoesFindLikelihoodRangeForMCMC) {
  {
    // This test verifies that FindLikelihoodRangeForMCMC accurately finds a
    // range where the quadratic gaussian log likelihood varies by +/- 2

    const double kMaximumPoint = 3.0;

    class NormalDensityFunc : public RealFunction {
     public:
      explicit NormalDensityFunc(double x) : max_(x) {}

      void Evaluate(ArraySlice<double> x,
                    MutableArraySlice<double> r) const override {
        for (int i = 0; i < x.size(); ++i) {
          r[i] = -(x[i] - max_) * (x[i] - max_) * 0.5;
        }
      }
     private:
      double max_;
    };

    NormalDensityFunc cback(kMaximumPoint);

    const int grid_size = 100;
    const pair<double, double>
        input_range(kMaximumPoint - 3, kMaximumPoint + 3);

    pair<double, double> out_range = FindLikelihoodRangeForMCMC(
        cback, kMaximumPoint /* initial_value */, grid_size,
        2.0 /* max_llik_delta */, input_range);
    EXPECT_NEAR(out_range.first, kMaximumPoint - 2, 0.04);
    EXPECT_NEAR(out_range.second, kMaximumPoint + 2, 0.04);
  }

  {
    static const double function_slope = 1.0;

    class IncreasingFunc : public RealFunction {
     public:
      void Evaluate(ArraySlice<double> x,
                    MutableArraySlice<double> r) const override {
        for (int i = 0; i < x.size(); ++i) {
          r[i] = function_slope * x[i];
        }
      }
    };

    IncreasingFunc cback;

    const double max_llik_delta = 2.0;
    const double initial_value = 3.0;
    const int grid_size = 100;
    const pair<double, double>
        input_range(initial_value - 3, initial_value + 3);

    pair<double, double> out_range = FindLikelihoodRangeForMCMC(
        cback, initial_value, grid_size,
        max_llik_delta, input_range);
    EXPECT_NEAR(out_range.first,
                initial_value - (max_llik_delta / function_slope),
                0.04);
    EXPECT_FLOAT_EQ(out_range.second, input_range.second);
  }

  {
    static const double function_slope = -0.5;

    class DecreasingFunc : public RealFunction {
     public:
      void Evaluate(ArraySlice<double> x,
                    MutableArraySlice<double> r) const override {
        for (int i = 0; i < x.size(); ++i) {
          r[i] = function_slope * x[i];
        }
      }
    };

    DecreasingFunc cback;

    const double max_llik_delta = 2.0;
    const double initial_value = 3.0;
    const int grid_size = 100;
    const pair<double, double>
        input_range(initial_value - 6, initial_value + 6);

    pair<double, double> out_range = FindLikelihoodRangeForMCMC(
        cback, initial_value, grid_size,
        max_llik_delta, input_range);
    EXPECT_FLOAT_EQ(out_range.first, input_range.first);
    EXPECT_NEAR(out_range.second,
                initial_value - (max_llik_delta / function_slope),
                0.1);
  }
}

TEST(GammaPriorOptimizeTest, DoesFitPrior) {
  // This dataset was generated by the following R code:
  // set.seed(15)
  // true.param <- 5
  // offsets <- rpois(100, 30) + 1
  // pois.params <- rgamma(length(offsets), true.param, true.param)
  // pois.obs <- rpois(length(offsets), offsets * pois.params)
  //
  // prior.eval <- seq(from = 0.01, to = 50, length = 10000)
  // prior.const <- prior.eval * log(prior.eval) - lgamma(prior.eval)
  // post.const <- rep(0, length(prior.eval))
  // for (k in seq_along(pois.obs)) {
  //   alpha <- prior.eval + pois.obs[k]
  //   beta <- prior.eval + offsets[k]
  //   post.const <- post.const + alpha * log(beta) - lgamma(alpha)
  // }
  //
  // marg.lik <- prior.const * length(offsets) - post.const
  // post.distn <- exp(marg.lik - max(marg.lik))
  // post.distn <- post.distn / sum(post.distn)
  //
  // param.lik <- sapply(prior.eval,
  //                     function(x) sum(dgamma(pois.params, x, x, log = TRUE)))
  //
  // sample.priors <- rmultinom(20000, size = 1, prob = post.distn)
  // sample.priors <- apply(sample.priors, 2, function(z) which(z > 0))
  // sample.priors <- prior.eval[sample.priors]
  //
  //
  // # print and plot the result
  // plot(prior.eval, marg.lik, type = "l", log = "x")
  // abline(v = true.param)
  //
  // range(prior.eval[marg.lik > max(marg.lik) - 3])  # 3.439657,  9.659115
  // prior.eval[which.max(marg.lik)]  # 5.694431
  // prior.eval[which.max(param.lik)]  # 5.844416
  //
  // sd(sample.priors)   # 0.9873732
  // mean(sample.priors) # 5.928602
  // quantile(sample.priors, c(0.001, 0.999)) # 3.439657 9.659115
  // paste(paste0("{", offsets, ", ", pois.obs, "}"), collapse = ", ")
  // paste(round(pois.params, 3), collapse = ", ")

  // A vector of pairs of (expected, actual) events
  vector<pair<double, double>> stats_data =
      {{32, 16}, {41, 50}, {29, 18}, {35, 18}, {33, 40}, {24, 16}, {33, 56},
       {36, 29}, {40, 51}, {35, 66}, {35, 54}, {22, 12}, {24, 31}, {30, 28},
       {27, 25}, {31, 28}, {24, 24}, {34, 43}, {30, 56}, {27, 29}, {28, 5},
       {18, 15}, {35, 21}, {27, 24}, {35, 42}, {27, 9}, {31, 27}, {38, 28},
       {35, 38}, {36, 28}, {33, 35}, {29, 44}, {36, 7}, {27, 23}, {34, 30},
       {37, 35}, {36, 23}, {38, 25}, {36, 34}, {35, 26}, {40, 33}, {36, 55},
       {34, 24}, {36, 37}, {29, 40}, {40, 42}, {33, 20}, {26, 36}, {25, 13},
       {37, 70}, {30, 37}, {32, 33}, {28, 41}, {23, 17}, {40, 26}, {35, 22},
       {22, 19}, {37, 13}, {45, 59}, {33, 22}, {34, 22}, {38, 74}, {34, 7},
       {38, 90}, {32, 50}, {33, 31}, {35, 30}, {33, 18}, {33, 38}, {36, 56},
       {39, 32}, {34, 54}, {36, 50}, {36, 38}, {28, 19}, {23, 15}, {24, 35},
       {29, 31}, {39, 40}, {30, 30}, {38, 47}, {35, 9}, {34, 24}, {34, 32},
       {38, 28}, {31, 15}, {30, 28}, {32, 35}, {31, 26}, {35, 17}, {23, 27},
       {34, 10}, {31, 29}, {33, 59}, {32, 51}, {34, 32}, {22, 39}, {45, 44},
       {29, 55}, {33, 53}};
  vector<double> stats_events(stats_data.size());
  vector<double> stats_p_events(stats_data.size());
  for (int i = 0; i < stats_data.size(); ++i) {
    stats_events[i] = stats_data[i].second;
    stats_p_events[i] = stats_data[i].first;
  }

  vector<double> coefficients =
      {0.376, 1.032, 0.547, 0.712, 1.184, 0.646, 1.704, 0.667, 1.3, 1.927,
       1.051, 0.664, 1.247, 0.857, 1.183, 0.849, 0.925, 1.324, 1.543, 1.108,
       0.258, 1.099, 0.642, 1.035, 1.19, 0.23, 0.921, 0.853, 0.986, 0.719,
       0.997, 1.477, 0.291, 0.894, 0.826, 1.126, 0.729, 0.78, 0.759, 0.591,
       0.98, 1.364, 0.79, 1.06, 1.382, 1.23, 0.48, 1.494, 0.659, 1.91, 1.557,
       0.899, 1.511, 0.868, 0.775, 0.814, 0.617, 0.453, 1.138, 0.898, 0.801,
       1.914, 0.273, 1.938, 1.28, 1.066, 0.607, 0.502, 1.192, 1.549, 0.627,
       1.692, 1.304, 1.343, 0.631, 0.68, 1.21, 1.133, 1.1, 0.905, 0.919,
       0.564, 0.648, 1.235, 0.762, 0.594, 0.889, 1.191, 1.166, 0.553, 1.184,
       0.503, 1.051, 1.557, 1.338, 0.884, 1.626, 0.91, 1.724, 1.473};

  static const double kIntegratedLikelihoodPrior = 5.694431;
  static const double kSampleLikelihoodPrior = 5.844416;
  static const double kSampleMeanPrior = 5.928602;
  static const double kSamplePriorStdDev = 0.9873732;
  static const pair<double, double> kSamplePriorQuantiles(3.439657,  9.659115);
  util::random::Distribution distn(15);

  FeatureFamilyPrior init_prior;
  init_prior.set_mean(1.0);
  init_prior.set_inverse_variance(20.0);

  {
    GammaPoissonPriorUpdater::PriorOptimConfig optim_config;
    optim_config.grid_size = 30;
    optim_config.num_iterations = 10;

    GammaPoissonPriorIntegratedOptimize prior_optim;
    prior_optim.SetPriorFromProto(init_prior);
    prior_optim.SetPriorOptimConfig(optim_config);
    prior_optim.UpdateVariance(stats_events, stats_p_events,
                               coefficients, &distn);

    FeatureFamilyPrior fitted_prior;
    prior_optim.SetProtoFromPrior(&fitted_prior);
    EXPECT_DOUBLE_EQ(fitted_prior.mean(), 1.0);
    EXPECT_NEAR(
        fitted_prior.inverse_variance(), kIntegratedLikelihoodPrior, 0.01);
  }

  {
    GammaPoissonPriorUpdater::PriorOptimConfig optim_config;
    optim_config.grid_size = 50;
    optim_config.num_iterations = 10;

    GammaPoissonPriorSampleOptimize prior_optim;
    prior_optim.SetPriorFromProto(init_prior);
    prior_optim.SetPriorOptimConfig(optim_config);
    prior_optim.UpdateVariance(stats_events, stats_p_events,
                               coefficients, &distn);

    FeatureFamilyPrior fitted_prior;
    prior_optim.SetProtoFromPrior(&fitted_prior);
    EXPECT_DOUBLE_EQ(fitted_prior.mean(), 1.0);
    EXPECT_NEAR(fitted_prior.inverse_variance(), kSampleLikelihoodPrior, 0.001);
  }

  {
    GammaPoissonPriorUpdater::PriorOptimConfig optim_config;
    optim_config.grid_size = 50;
    optim_config.num_iterations = 10;

    GammaPoissonPriorRaoBlackwellizedOptimize prior_optim;
    prior_optim.SetPriorFromProto(init_prior);
    prior_optim.SetPriorOptimConfig(optim_config);

    // This update depends on the prior currently held by 'prior_optim' so it
    // takes multiple iterations to converge
    for (int i = 0; i < 30; ++i) {
      prior_optim.UpdateVariance(stats_events, stats_p_events,
                                 coefficients, &distn);
    }

    FeatureFamilyPrior fitted_prior;
    prior_optim.SetProtoFromPrior(&fitted_prior);
    EXPECT_DOUBLE_EQ(fitted_prior.mean(), 1.0);
    EXPECT_NEAR(
        fitted_prior.inverse_variance(), kIntegratedLikelihoodPrior, 0.005);
  }

  {
    GammaPoissonPriorUpdater::PriorOptimConfig optim_config;
    optim_config.grid_size = 10;
    optim_config.num_iterations = 50;

    GammaPoissonPriorIntegratedGibbsSampler prior_optim;
    prior_optim.SetPriorFromProto(init_prior);
    prior_optim.SetPriorOptimConfig(optim_config);

    vector<double> sample_priors;
    for (int i = 0; i < 30; ++i) {
      prior_optim.UpdateVariance(stats_events, stats_p_events,
                                 coefficients, &distn);

      FeatureFamilyPrior fitted_prior;
      prior_optim.SetProtoFromPrior(&fitted_prior);
      EXPECT_DOUBLE_EQ(fitted_prior.mean(), 1.0);
      sample_priors.push_back(fitted_prior.inverse_variance());
    }

    double sum = 0.0;
    double sum_squared = 0.0;
    for (double x : sample_priors) {
      sum += x;
      sum_squared += x * x;
      CHECK_GT(x, kSamplePriorQuantiles.first);
      CHECK_LT(x, kSamplePriorQuantiles.second);
    }
    const double sample_mean = sum / static_cast<double>(sample_priors.size());
    const double sample_var =
        (sum_squared / static_cast<double>(sample_priors.size()))
        - sample_mean * sample_mean;
    const double sample_sd = sqrt(sample_var);
    EXPECT_NEAR(sample_mean, kSampleMeanPrior, 0.5);
    EXPECT_NEAR(sample_sd, kSamplePriorStdDev, 0.1);
  }
}

}  // namespace emre
