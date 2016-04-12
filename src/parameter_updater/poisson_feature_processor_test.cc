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

#include "poisson_feature_processor.h"  // NOLINT

#include <algorithm>

#include "contentads/analysis/caa/search_plus/regmh/emre/src/indexers/bias_indexer.h"
#include "contentads/analysis/caa/search_plus/regmh/emre/src/indexers/memory_indexer.h"
#include "testing/base/public/gunit.h"

namespace {

using emre::BiasIndexReader;
using emre::IndexReader;
using emre::MemoryIndexBuilder;
using emre::PoissonFeatureProcessor;
using emre::PoissonScaledFeatureProcessor;
using emre::UpdateProcessor;

class PoissonFeatureProcessorTest : public ::testing::Test { };

TEST_F(PoissonFeatureProcessorTest, DoesGetStatsForUpdateHighMultiplicity) {
  // This test verifies that MemoryIndex does job Foo.
  static const int kNumData = 500;
  static const int kNumLevels = 500;
  static const double kNumEvents = 3.0;
  static const double kOffset = 1.0;
  static const double kPrediction = 2.0;

  vector<double> response(kNumData, kNumEvents);
  vector<double> offsets(kNumData, kOffset);
  vector<double> coefficients(kNumLevels, kOffset);
  vector<double> p_events(kNumData, kPrediction);
  vector<double> predicted(kNumLevels, 0.0);

  {
    static const int kSizeHint = 0;
    MemoryIndexBuilder idx(string("test_feature"), kSizeHint);
    for (int i = 0; i < kNumData; ++i) {
      idx.GetInt64LevelId(i);
      idx.WriteLevelId(i);
    }

    std::unique_ptr<IndexReader> index = idx.MoveToReader();

    PoissonFeatureProcessor processor;
    // Before calling GetStatsForUpdate, we need to initialize predicted
    // with the immutable part of the sufficient statistic
    std::fill(predicted.begin(), predicted.end(), 0.0);
    processor.GetStatsForUpdate(
        index.get(), offsets, coefficients, p_events, &predicted);

    for (int i = 0; i < kNumData; ++i) {
      EXPECT_FLOAT_EQ(predicted[i], kPrediction);
    }
  }

  {
    static const int kSizeHint = 1 << 20;
    MemoryIndexBuilder idx(string("test_feature"), kSizeHint);
    for (int i = 0; i < kNumData; ++i) {
      idx.GetInt64LevelId(i);
      idx.WriteLevelId(i);
    }

    std::unique_ptr<IndexReader> index = idx.MoveToReader();

    PoissonFeatureProcessor processor;
    // Reset predictions to reuse the storgage
    predicted.assign(kNumLevels, 0.0);

    processor.GetStatsForUpdate(
        index.get(), offsets, coefficients, p_events, &predicted);

    for (int i = 0; i < kNumData; ++i) {
      EXPECT_FLOAT_EQ(predicted[i], kPrediction);
    }
  }
}


TEST_F(PoissonFeatureProcessorTest, DoesGetStatsForUpdateLowMultiplicity) {
  static const int kNumData = 500;
  static const double kNumDataDbl = static_cast<double>(kNumData);
  static const int kNumLevels = 1;
  static const double kNumEvents = 3.0;
  static const double kOffset = 1.0;
  static const double kPrediction = 2.0;

  vector<double> response(kNumData, kNumEvents);
  vector<double> offsets(kNumData, kOffset);
  vector<double> coefficients(kNumLevels, kOffset);
  vector<double> p_events(kNumData, kPrediction);
  vector<double> predicted(kNumLevels, 0.0);

  static const int kSizeHint = 0;
  MemoryIndexBuilder idx(string("test_feature"), kSizeHint);
  for (int i = 0; i < kNumData; ++i) {
    int seq_index = (i % 2) - 1;
    if (seq_index != -1) idx.GetInt64LevelId(seq_index);
    idx.WriteLevelId(seq_index);
  }
  EXPECT_EQ(idx.NextLevelId(), 1);

  std::unique_ptr<IndexReader> index = idx.MoveToReader();

  PoissonFeatureProcessor processor;
  // Before calling GetStatsForUpdate, we need to initialize predicted
  // with the immutable part of the sufficient statistic
  std::fill(predicted.begin(), predicted.end(), 0.0);
  processor.GetStatsForUpdate(
      index.get(), offsets, coefficients, p_events, &predicted);

  EXPECT_FLOAT_EQ(predicted[0], kPrediction * kNumDataDbl / 2);
}


TEST_F(PoissonFeatureProcessorTest, DoesAddToPrediction) {
  // This test verifies that the base class PoissonFeatureUpdater
  // performs AddToPrediction correctly
  static const int kNumData = 4;

  {
    vector<double> p_events = { 1.0, 1.0, 0.5, 0.1 };
    vector<double> coefficients = { 2.0, 3.0 };
    static const int kSizeHint = 0;
    MemoryIndexBuilder idx(string("test_feature"), kSizeHint);
    for (int i = 0; i < kNumData; ++i) {
      int seq_index = i % 2;
      idx.GetInt64LevelId(seq_index);
      idx.WriteLevelId(seq_index);
    }

    std::unique_ptr<IndexReader> index = idx.MoveToReader();
    PoissonFeatureProcessor processor;
    processor.AddToPrediction(index.get(), coefficients, &p_events);
    EXPECT_FLOAT_EQ(p_events[0], 2.0);
    EXPECT_FLOAT_EQ(p_events[1], 3.0);
    EXPECT_FLOAT_EQ(p_events[2], 1.0);
    EXPECT_FLOAT_EQ(p_events[3], 0.3);
  }
}


TEST_F(PoissonFeatureProcessorTest, DoesUpdatePredictions) {
  // This test verifies that the base class PoissonFeatureUpdater
  // performs UpdatePredictions correctly.
  const int kNumData = 4;
  static const int kSizeHint = 0;
  MemoryIndexBuilder idx(string("test_feature"), kSizeHint);
  for (int i = 0; i < kNumData; ++i) {
    int seq_index = i % 2;
    idx.GetInt64LevelId(seq_index);
    idx.WriteLevelId(seq_index);
  }

  vector<double> p_events = { 1.0, 1.0, 0.5, 0.1 };
  vector<double> coefficient_changes = { 2.0, 0.1 };

  std::unique_ptr<IndexReader> index = idx.MoveToReader();
  PoissonFeatureProcessor processor;
  processor.UpdatePredictions(index.get(), coefficient_changes, &p_events);
  EXPECT_FLOAT_EQ(p_events[0], 2.0);
  EXPECT_FLOAT_EQ(p_events[1], 0.1);
  EXPECT_FLOAT_EQ(p_events[2], 1.0);
  EXPECT_FLOAT_EQ(p_events[3], 0.01);
}

class PoissonBiasProcessorTest : public ::testing::Test { };

TEST_F(PoissonBiasProcessorTest, BiasIndexDoesGetStatsForUpdate) {
  // This test verifies that BiasIndex correctly implements
  // GetStatsForUpdate
  static const int kNumData = 500;
  static const double kNumDataDbl = static_cast<double>(kNumData);
  static const int kNumLevels = 1;
  static const double kNumEvents = 3.0;
  static const double kOffset = 1.0;
  static const double kPrediction = 2.0;

  vector<double> response(kNumData, kNumEvents);
  vector<double> offsets(kNumData, kOffset);
  vector<double> coefficients(kNumLevels, kOffset);
  vector<double> p_events(kNumData, kPrediction);
  vector<double> predicted(kNumLevels, 0.0);
  BiasIndexReader bias_index(kNumData);

  PoissonFeatureProcessor processor;
  // Before calling GetStatsForUpdate, we need to initialize predicted
  // with the immutable part of the sufficient statistic
  predicted[0] = 0.0;
  processor.GetStatsForUpdate(
      &bias_index, offsets, coefficients, p_events, &predicted);

  EXPECT_FLOAT_EQ(predicted[0], kPrediction * kNumDataDbl);
}

TEST_F(PoissonBiasProcessorTest, DoesAddToPrediction) {
  // This test verifies that FeatureIndex and its subclasses BiasIndex and
  // OnePerRowIndex perform AddToPrediction correctly
  static const int kNumData = 4;
  {
    vector<double> p_events = { 1.0, 1.0, 0.5, 0.1 };
    vector<double> coefficients = { 3.0 };
    BiasIndexReader bias_index(kNumData);

    PoissonFeatureProcessor processor;
    processor.AddToPrediction(&bias_index, coefficients, &p_events);
    EXPECT_FLOAT_EQ(p_events[0], 3.0);
    EXPECT_FLOAT_EQ(p_events[1], 3.0);
    EXPECT_FLOAT_EQ(p_events[2], 1.5);
    EXPECT_FLOAT_EQ(p_events[3], 0.3);
  }
}

class PoissonScaledFeatureProcessorTest : public ::testing::Test { };

TEST_F(PoissonScaledFeatureProcessorTest, DoesGetStatsForUpdate) {
  // Set up the mock data, with 5 levels including scalings
  const vector<pair<int, double>> lev_scaling =
      {{0, 1.0}, {-1, 0.0}, {0, 1.0}, {1, 2.0}, {1, 2.0}, {1, 3.0}, {1, 2.0},
       {0, -1.0}, {2, 3.5}};
  const int kNumData = lev_scaling.size();

  // Set up the feature indexer
  MemoryIndexBuilder builder("test_feature", 0 /* size_hint */);
  for (int i = 0; i < kNumData; ++i) {
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
  std::unique_ptr<IndexReader> index = builder.MoveToReader();

  PoissonScaledFeatureProcessor processor(index.get());
  // Check the aggregate scaling
  UpdateProcessor::SupplementalStats supp_stats;
  processor.PrepareUpdater(&supp_stats);
  EXPECT_EQ(supp_stats.aggregate_scaling.size(), 5);
  EXPECT_DOUBLE_EQ(supp_stats.aggregate_scaling[0], 1.0);
  EXPECT_DOUBLE_EQ(supp_stats.aggregate_scaling[1], -1.0);
  EXPECT_DOUBLE_EQ(supp_stats.aggregate_scaling[2], 2.0);
  EXPECT_DOUBLE_EQ(supp_stats.aggregate_scaling[3], 3.0);
  EXPECT_DOUBLE_EQ(supp_stats.aggregate_scaling[4], 3.5);

  const int num_levels = processor.GetStatsSize(0);
  EXPECT_EQ(num_levels, 5);
  const vector<double> coefficients = {1.0, 2.0, -3.0};
  const vector<double> p_events(kNumData, 1.0);
  const vector<double> offsets(kNumData, 0.0);
  // properly pre-allocated predicted stats
  vector<double> predicted(num_levels, 0.0);

  processor.GetStatsForUpdate(
      index.get(), offsets, coefficients, p_events, &predicted);

  // predicted events for 1st aggregate level {0, 1.0} occurs twice
  EXPECT_DOUBLE_EQ(predicted[0],
                   2 * exp(-lev_scaling[0].second * coefficients[0]));
  // predicted events for 2nd aggregate level {0, -1.0} occurs once
  EXPECT_DOUBLE_EQ(predicted[1],
                   exp(-lev_scaling[7].second * coefficients[0]));
  // predicted events for 3rd aggregate level {1, 2.0} occurs trice
  EXPECT_DOUBLE_EQ(predicted[2],
                   3 * exp(-lev_scaling[3].second * coefficients[1]));
  // predicted events for 4th aggregate level {1, 3.0} occurs once
  EXPECT_DOUBLE_EQ(predicted[3],
                   exp(-lev_scaling[5].second * coefficients[1]));
  // predicted events for 5th aggregate level {2, 3.5} occurs once
  EXPECT_DOUBLE_EQ(predicted[4],
                   exp(-lev_scaling[8].second * coefficients[2]));
}

TEST_F(PoissonScaledFeatureProcessorTest, DoesAddToPrediction) {
  // Set up the mock data, with 5 levels including scalings
  const vector<pair<int, double>> lev_scaling =
      {{0, 1.0}, {-1, 0.0}, {0, 1.0}, {1, 2.0}, {1, 2.0}, {1, 3.0}, {1, 2.0},
       {0, -1.0}, {2, 3.5}};
  const int kNumData = lev_scaling.size();

  // Set up the feature indexer
  MemoryIndexBuilder builder("test_feature", 0 /* size_hint */);
  for (int i = 0; i < kNumData; ++i) {
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
  std::unique_ptr<IndexReader> index = builder.MoveToReader();

  PoissonScaledFeatureProcessor processor(index.get());

  const vector<double> coefficients = {1.0, 2.0, -3.0};

  // Initialize p.events with offsets, here 1.0 ... kNumData
  vector<double> p_events(kNumData);
  for (int i = 0; i < kNumData; ++i) {
    p_events[i] = 2 * static_cast<double>(i);
  }
  processor.AddToPrediction(index.get(), coefficients, &p_events);

  // Check p_events
  for (int i = 0; i < p_events.size(); ++i) {
    int level_index = lev_scaling[i].first;
    if (level_index < 0) {
      EXPECT_DOUBLE_EQ(p_events[i], 2 * i);
    } else {
      double scaling = lev_scaling[i].second;
      EXPECT_DOUBLE_EQ(p_events[i],
                       2 * i * exp(scaling * coefficients[level_index]));
    }
  }
}

TEST_F(PoissonScaledFeatureProcessorTest, DoesUpdatePredictions) {
  // Set up the mock data, with 5 levels including scalings
  const vector<pair<int, double>> lev_scaling =
      {{0, 1.0}, {-1, 0.0}, {0, 1.0}, {1, 2.0}, {1, 2.0}, {1, 3.0}, {1, 2.0},
       {0, -1.0}, {2, 3.5}};
  const int kNumData = lev_scaling.size();

  // Set up the feature indexer
  MemoryIndexBuilder builder("test_feature", 0 /* size_hint */);
  for (int i = 0; i < kNumData; ++i) {
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
  std::unique_ptr<IndexReader> index = builder.MoveToReader();

  PoissonScaledFeatureProcessor processor(index.get());

  const vector<double> coefficient_changes = {1.0, 2.0, -3.0};

  // Initialize p.events with offsets, here 1.0 ... kNumData
  vector<double> p_events(kNumData);
  vector<double> new_p_events(kNumData);
  for (int i = 0; i < kNumData; ++i) {
    p_events[i] = 2 * static_cast<double>(i);
    new_p_events[i] = p_events[i];
  }
  processor.UpdatePredictions(index.get(), coefficient_changes, &new_p_events);

  // Check new_p_events
  for (int i = 0; i < p_events.size(); ++i) {
    int level_index = lev_scaling[i].first;
    if (level_index < 0) {
      EXPECT_DOUBLE_EQ(new_p_events[i], p_events[i]);
    } else {
      double scaling = lev_scaling[i].second;
      EXPECT_DOUBLE_EQ(new_p_events[i],
        p_events[i] * exp(scaling * coefficient_changes[level_index]));
    }
  }
}

}  // namespace
