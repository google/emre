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

#include "scaled_feature_util.h"  // NOLINT

#include "contentads/analysis/caa/search_plus/regmh/emre/src/indexers/memory_indexer.h"
#include "contentads/analysis/caa/search_plus/regmh/emre/src/indexers/memory_vector.h"
#include "contentads/analysis/caa/search_plus/regmh/emre/src/indexers/vector_storage.h"
#include "testing/base/public/gunit.h"

namespace emre {

TEST(ScaledFeatureUtilTest, DoesCreateMappings) {
  const vector<pair<int, double>> lev_scaling =
      {{0, 1.0}, {-1, 0.0}, {0, 1.0}, {1, 2.0}, {1, 2.0}, {1, 3.0}, {1, 2.0},
       {0, -1.0}, {2, 0.1}};

  MemoryIndexBuilder builder("test_feature", 0 /* size_hint */);
  for (int i = 0; i < lev_scaling.size(); ++i) {
    if (lev_scaling[i].first == -1) {
      builder.WriteLevelId(-1);  // add this feature as missing
      builder.MayWriteScaling(1.0);  // default scaling
      continue;
    }
    const int val = builder.GetInt64LevelId(lev_scaling[i].first);
    EXPECT_EQ(val, lev_scaling[i].first);
    builder.WriteLevelId(val);
    builder.MayWriteScaling(lev_scaling[i].second);
  }

  std::unique_ptr<IndexReader> reader = builder.MoveToReader();

  vector<pair<int, int>> level_posn_size;
  MemoryVectorBuilder<int> level_scaling_posn_builder;
  vector<double> aggregate_scaling;
  ScaledFeatureUtil::MakeLevelScalingMapping(
      reader.get(), &level_posn_size, &level_scaling_posn_builder,
      &aggregate_scaling);

  std::unique_ptr<VectorReader<int>> level_scaling_posn
      = level_scaling_posn_builder.MoveToReader();
  const auto& level_scaling_posn_vec = level_scaling_posn->AsVectorForTesting();

  EXPECT_EQ(level_posn_size.size(), 3);
  EXPECT_EQ(level_scaling_posn_vec.size(), 9);
  EXPECT_EQ(aggregate_scaling.size(), 5);

  EXPECT_EQ(level_posn_size[0].first, 0);
  EXPECT_EQ(level_posn_size[0].second, 2);
  EXPECT_EQ(level_posn_size[1].first, 2);
  EXPECT_EQ(level_posn_size[1].second, 2);
  EXPECT_EQ(level_posn_size[2].first, 4);
  EXPECT_EQ(level_posn_size[2].second, 1);

  EXPECT_EQ(aggregate_scaling[0], 1.0);
  EXPECT_EQ(aggregate_scaling[1], -1.0);
  EXPECT_EQ(aggregate_scaling[2], 2.0);
  EXPECT_EQ(aggregate_scaling[3], 3.0);
  EXPECT_EQ(aggregate_scaling[4], 0.1);

  EXPECT_EQ((level_scaling_posn_vec)[0], 0);
  EXPECT_EQ((level_scaling_posn_vec)[1], -1);
  EXPECT_EQ((level_scaling_posn_vec)[2], 0);
  EXPECT_EQ((level_scaling_posn_vec)[3], 2);
  EXPECT_EQ((level_scaling_posn_vec)[4], 2);
  EXPECT_EQ((level_scaling_posn_vec)[5], 3);
  EXPECT_EQ((level_scaling_posn_vec)[6], 2);
  EXPECT_EQ((level_scaling_posn_vec)[7], 1);
  EXPECT_EQ((level_scaling_posn_vec)[8], 4);
}

TEST(ScaledFeatureUtilTest, DoesGetStatsForUpdate) {
  const vector<pair<int, double>> lev_scaling =
      {{0, 1.0}, {-1, 0.0}, {0, 1.0}, {1, 2.0}, {1, 2.0}, {1, 3.0}, {1, 2.0},
       {0, -1.0}, {2, 0.1}};
  const vector<double> p_events(lev_scaling.size(), 1.0);
  const vector<double> coefficients = {1.0, -1.0, 2.0};

  vector<double> events(lev_scaling.size(), 2.0);
  events[lev_scaling.size() - 1] = 1.0;

  MemoryIndexBuilder builder("test_feature", 0 /* size_hint */);
  for (int i = 0; i < lev_scaling.size(); ++i) {
    if (lev_scaling[i].first == -1) {
      builder.WriteLevelId(-1);  // add this feature as missing
      builder.MayWriteScaling(1.0);  // default scaling
      continue;
    }
    const int val = builder.GetInt64LevelId(lev_scaling[i].first);
    EXPECT_EQ(val, lev_scaling[i].first);
    builder.WriteLevelId(val);
    builder.MayWriteScaling(lev_scaling[i].second);
  }
  std::unique_ptr<IndexReader> index_reader = builder.MoveToReader();

  vector<pair<int, int>> level_posn_size;
  MemoryVectorBuilder<int> level_scaling_posn_builder;
  vector<double> aggregate_scaling;
  ScaledFeatureUtil::MakeLevelScalingMapping(
      index_reader.get(), &level_posn_size, &level_scaling_posn_builder,
      &aggregate_scaling);

  std::unique_ptr<VectorReader<int>> level_scaling_posn
      = level_scaling_posn_builder.MoveToReader();

  vector<double> events_stats(aggregate_scaling.size(), 0.0);
  vector<double> p_events_stats(aggregate_scaling.size(), 0.0);

  // Assumes that 'events_stats' & 'p_events_stats' is already allocated
  // and zero'ed out
  ScaledFeatureUtil::GetEventsForPoissonUpdate(
      index_reader.get(), level_scaling_posn.get(), events, &events_stats);
  ScaledFeatureUtil::GetPredictionForPoissonUpdate(
      index_reader.get(), level_scaling_posn.get(), coefficients, p_events,
      &p_events_stats);

  EXPECT_NEAR(p_events_stats[0], 2.0 * exp(-1.0), 1e-6);
  EXPECT_NEAR(events_stats[0], 4.0, 1e-6);
  EXPECT_NEAR(p_events_stats[1], exp(1.0), 1e-6);
  EXPECT_NEAR(events_stats[1], 2.0, 1e-6);
  EXPECT_NEAR(p_events_stats[2], 3.0 * exp(2.0), 1e-6);
  EXPECT_NEAR(events_stats[2], 6.0, 1e-6);
  EXPECT_NEAR(p_events_stats[3], 1.0 * exp(3.0), 1e-6);
  EXPECT_NEAR(events_stats[3], 2.0, 1e-6);
  EXPECT_NEAR(p_events_stats[4], 1.0 * exp(-0.1 * 2.0), 1e-6);
  EXPECT_NEAR(events_stats[4], 1.0, 1e-6);
}

TEST(ScaledFeatureUtilTest, DoesMCMCSample) {
  const vector<pair<int, double>> lev_scaling =
      {{0, 1.0}, {-1, 0.0}, {0, 1.0}, {1, 2.0}, {1, 2.0}, {1, 3.0}, {1, 2.0},
       {0, -1.0}, {2, 30.0}};
  const vector<double> events =
      {1000.0 / 2.0, 0.01, 1000.0 / 2.0, 4.0 / 3.0, 4.0 / 3.0, 8.0, 4.0 / 3.0,
       10000.0, 1.0};
  const vector<double> p_events =
      {10000.0 / 2.0, 0.001, 10000.0 / 2.0, 1.0 / 3.0, 1.0 / 3.0, 1.0,
       1.0 / 3.0, 1000.0, 1.0};

  // the stats data
  const vector<double> predicted_stats = {10000.0, 1000.0, 1.0, 1.0, 1.0};
  const vector<double> events_stats = {1000.0, 10000.0, 4.0, 8.0, 1.0};

  MemoryIndexBuilder builder("test_feature", 0 /* size_hint */);
  for (int i = 0; i < lev_scaling.size(); ++i) {
    if (lev_scaling[i].first == -1) {
      builder.WriteLevelId(-1);  // add this feature as missing
      builder.MayWriteScaling(1.0);  // default scaling
      continue;
    }
    const int val = builder.GetInt64LevelId(lev_scaling[i].first);
    EXPECT_EQ(val, lev_scaling[i].first);
    builder.WriteLevelId(val);
    builder.MayWriteScaling(lev_scaling[i].second);
  }

  std::unique_ptr<IndexReader> reader = builder.MoveToReader();

  vector<pair<int, int>> level_posn_size;
  MemoryVectorBuilder<int> level_scaling_posn_builder;
  vector<double> aggregate_scaling;
  ScaledFeatureUtil::MakeLevelScalingMapping(
      reader.get(), &level_posn_size, &level_scaling_posn_builder,
      &aggregate_scaling);

  std::unique_ptr<VectorReader<int>> level_scaling_posn
      = level_scaling_posn_builder.MoveToReader();

  vector<double> agg_events(aggregate_scaling.size(), 0.0);
  vector<double> predicted(aggregate_scaling.size(), 0.0);

  const vector<double> zero_coefficients = {0.0, 0.0, 0.0};
  // Assumes that 'stats' is already allocated and zero'ed out
  ScaledFeatureUtil::GetPredictionForPoissonUpdate(
      reader.get(), level_scaling_posn.get(), zero_coefficients,
      p_events, &predicted);

  ScaledFeatureUtil::GetEventsForPoissonUpdate(
      reader.get(), level_scaling_posn.get(), events, &agg_events);

  for (int i = 0; i < predicted.size(); ++i) {
    EXPECT_NEAR(predicted_stats[i], predicted[i], 0.001);
    EXPECT_NEAR(events_stats[i], agg_events[i], 0.001);
  }

  util::random::Distribution distn(15);

  const int num_iterations = 1000;
  const int num_burnin = 200;
  const int num_steps_per_iteration = 20;

  const double prior_inverse_variance = 1.0 / (2.0 * 2.0);

  // First we run the MCMC on the aggregated statistics
  {
    vector<double> ranefs = {0.0, 0.0, 1.0};
    vector<double> proposal_sds = {5.0, 5.0, 5.0};
    vector<pair<int, int>> acceptance_counts = {{0, 0}, {0, 0}, {0, 0}};

    vector<double> ranefs_sum = {0.0, 0.0, 0.0};
    vector<double> ranefs_sum_sqr = {0.0, 0.0, 0.0};
    for (int iter_idx = 0; iter_idx < num_iterations; ++iter_idx) {
      poisson::ScaledFeatureMhProposer proposer(
          level_posn_size, aggregate_scaling,
          predicted_stats, events_stats,
          proposal_sds, &distn);

      poisson::ScaledFeatureLoglik llik_cback(
          prior_inverse_variance, level_posn_size, aggregate_scaling,
          predicted_stats, events_stats);

      for (int i = 0; i < ranefs.size(); ++i) {
        int num_accepted = 0;
        ranefs[i] = emre::metropolis_hastings::RunMetropolisHastings(
            llik_cback, ranefs[i], num_steps_per_iteration, &proposer, &distn,
            &num_accepted);

        if (iter_idx >= num_burnin) {
          ranefs_sum[i] += ranefs[i];
          ranefs_sum_sqr[i] += ranefs[i] * ranefs[i];
        }

        proposer.IncrementLevelIndex();
        llik_cback.IncrementLevelIndex();

        auto* ct = &(acceptance_counts[i]);
        ct->first += num_steps_per_iteration;
        ct->second += num_accepted;

        if (ct->second <= 0.2 * ct->first) {
          proposal_sds[i] *= 0.5;
          LOG(INFO) << "reducing step size for ranef " << i << " to "
              << proposal_sds[i];
          ct->first = 0;
          ct->second = 0;
        } else if (ct->second >= 0.8 * ct->first) {
          proposal_sds[i] *= 2.0;
          LOG(INFO) << "increasing step size for ranef " << i << " to "
              << proposal_sds[i];
          ct->first = 0;
          ct->second = 0;
        }
      }
    }

    const double scl = 1.0 / (num_iterations - num_burnin);
    vector<double> ranefs_mean = {ranefs_sum[0] * scl, ranefs_sum[1] * scl,
                                  ranefs_sum[2] * scl};
    vector<double> ranefs_var =
        {ranefs_sum_sqr[0] * scl - ranefs_mean[0] * ranefs_mean[0],
         ranefs_sum_sqr[1] * scl - ranefs_mean[1] * ranefs_mean[1],
         ranefs_sum_sqr[2] * scl - ranefs_mean[2] * ranefs_mean[2]};

    // These expected values are the known 'true' values.  The acceptable errors
    // in the third argument were determined empirically by looking at the
    // result of this unit test.
    EXPECT_NEAR(ranefs_mean[0], log(0.1), 0.01);
    EXPECT_NEAR(ranefs_mean[1], log(2.0), 0.1);
    EXPECT_NEAR(ranefs_mean[2], 0.0, 0.1);

    // These expected values are found empirically be looking at the result of
    // this unit test.
    EXPECT_NEAR(sqrt(ranefs_var[0]), 0.00920835, 0.01);
    EXPECT_NEAR(sqrt(ranefs_var[1]), 0.107063, 0.03);
    EXPECT_NEAR(sqrt(ranefs_var[2]), 0.0424447, 0.01);
    EXPECT_NEAR(proposal_sds[0], 0.0390625, 0.03);
    EXPECT_NEAR(proposal_sds[1], 0.3125, 0.03);
    EXPECT_NEAR(proposal_sds[2], 0.078125, 0.03);
  }

  // Now we run the MCMC on unaggregated statistics and check for the same
  // result
  {
    vector<double> ranefs = {0.0, 0.0, 1.0};
    vector<double> proposal_sds = {5.0, 5.0, 5.0};
    vector<pair<int, int>> acceptance_counts = {{0, 0}, {0, 0}, {0, 0}};


    vector<double> ranefs_sum = {0.0, 0.0, 0.0};
    vector<double> ranefs_sum_sqr = {0.0, 0.0, 0.0};
    for (int iter_idx = 0; iter_idx < num_iterations; ++iter_idx) {
      vector<int> num_accept(ranefs.size(), 0);
      ScaledFeatureUtil::GaussianProposalMHPoissonUpdate(
        num_steps_per_iteration, prior_inverse_variance, reader.get(),
        events, p_events, ranefs, proposal_sds,
        &ranefs, &num_accept, &distn);

      for (int i = 0; i < ranefs.size(); ++i) {
        acceptance_counts[i].first += num_steps_per_iteration;
        acceptance_counts[i].second += num_accept[i];

        if (iter_idx >= num_burnin) {
          ranefs_sum[i] += ranefs[i];
          ranefs_sum_sqr[i] += ranefs[i] * ranefs[i];
        }

        auto* ct = &(acceptance_counts[i]);
        if (ct->second <= 0.2 * ct->first) {
          proposal_sds[i] *= 0.5;
          LOG(INFO) << "reducing step size for ranef " << i << " to "
              << proposal_sds[i];
          ct->first = 0;
          ct->second = 0;
        } else if (ct->second >= 0.8 * ct->first) {
          proposal_sds[i] *= 2.0;
          LOG(INFO) << "increasing step size for ranef " << i << " to "
              << proposal_sds[i];
          ct->first = 0;
          ct->second = 0;
        }
      }
    }

    const double scl = 1.0 / (num_iterations - num_burnin);
    vector<double> ranefs_mean = {ranefs_sum[0] * scl, ranefs_sum[1] * scl,
                                  ranefs_sum[2] * scl};
    vector<double> ranefs_var =
        {ranefs_sum_sqr[0] * scl - ranefs_mean[0] * ranefs_mean[0],
         ranefs_sum_sqr[1] * scl - ranefs_mean[1] * ranefs_mean[1],
         ranefs_sum_sqr[2] * scl - ranefs_mean[2] * ranefs_mean[2]};

    EXPECT_NEAR(ranefs_mean[0], log(0.1), 0.01);
    EXPECT_NEAR(ranefs_mean[1], log(2.0), 0.1);
    EXPECT_NEAR(ranefs_mean[2], 0.0, 0.1);
    EXPECT_NEAR(sqrt(ranefs_var[0]), 0.00920835, 0.01);
    EXPECT_NEAR(sqrt(ranefs_var[1]), 0.107063, 0.03);
    EXPECT_NEAR(sqrt(ranefs_var[2]), 0.0424447, 0.01);
    EXPECT_NEAR(proposal_sds[0], 0.0390625, 0.03);
    EXPECT_NEAR(proposal_sds[1], 0.3125, 0.03);
    EXPECT_NEAR(proposal_sds[2], 0.15625, 0.03);
  }
}

TEST(ScaledFeatureUtilTest, RootFindingSucceds) {
  const double prior_inverse_variance = 0.0;

  vector<double> events_stats = {50};
  vector<double> predicted_stats = {0.1};
  vector<pair<int, int>> level_posn_size;
  level_posn_size.emplace_back(0, 1);
  {
    // one positive scaling level 0.5, dlLik/dx = 0.5 (50 - 0.1 Exp[0.5 x])
    vector<double> aggregate_scaling = {0.5};

    poisson::ScaledFeatureLoglik llik_cback(
        prior_inverse_variance, level_posn_size, aggregate_scaling,
        predicted_stats, events_stats);
    poisson::ScaledFeatureRootSolver solver(&llik_cback);

    llik_cback.IncrementLevelIndex();
    double x0;
    int steps = solver.Solve(&x0);
    EXPECT_NEAR(x0, 12.4292, 0.01);
    EXPECT_LT(steps, 6);
  }
  {
    // one negative scaling level -0.5, dlLik/dx = -0.5 (50 - 0.1 Exp[-0.5 x])
    vector<double> aggregate_scaling = {-0.5};

    poisson::ScaledFeatureLoglik llik_cback(
        prior_inverse_variance, level_posn_size, aggregate_scaling,
        predicted_stats, events_stats);
    poisson::ScaledFeatureRootSolver solver(&llik_cback);

    llik_cback.IncrementLevelIndex();
    double x0;
    int steps = solver.Solve(&x0);
    EXPECT_NEAR(x0, -12.4292, 0.01);
    EXPECT_LT(steps, 6);
  }
}

TEST(ScaledFeatureUtilTest, FindRootAtBoundary) {
  const double prior_inverse_variance = 0.0;
  vector<double> events_stats = {50, 50};
  vector<double> predicted_stats = {0.1, 0.1};
  vector<pair<int, int>> level_posn_size;
  level_posn_size.emplace_back(0, 2);
  {
    // symmetric positive & negative scaling level -0.5 0.5 with,
    // dlLik/dx = -0.5 (50 - 0.1 Exp[-0.5 x]) + 0.5 (50 - 0.1 Exp[0.5 x])
    vector<double> aggregate_scaling = {-0.5, 0.5};

    poisson::ScaledFeatureLoglik llik_cback(
        prior_inverse_variance, level_posn_size, aggregate_scaling,
        predicted_stats, events_stats);
    poisson::ScaledFeatureRootSolver solver(&llik_cback);

    llik_cback.IncrementLevelIndex();
    double x0 = 0.0;
    int steps0 = solver.Solve(&x0);
    EXPECT_NEAR(x0, 0, 0.01);
    EXPECT_EQ(steps0, 0);

    double x1 = -0.05;
    int steps1 = solver.Solve(&x1);
    EXPECT_NEAR(x1, 0, 0.01);
    EXPECT_EQ(steps1, 0);

    double x2 = 0.05;
    int steps2 = solver.Solve(&x2);
    EXPECT_NEAR(x1, 0, 0.01);
    EXPECT_EQ(steps2, 0);
  }
}

}  // namespace emre
