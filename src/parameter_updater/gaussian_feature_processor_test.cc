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

#include "gaussian_feature_processor.h"  // NOLINT

#include "contentads/analysis/caa/search_plus/regmh/emre/src/indexers/bias_indexer.h"
#include "contentads/analysis/caa/search_plus/regmh/emre/src/indexers/memory_indexer.h"
#include "testing/base/public/gunit.h"

namespace {

using emre::BiasIndexReader;
using emre::FeatureData;
using emre::GaussianFeatureProcessor;
using emre::IndexReader;
using emre::MemoryIndexBuilder;

class GaussianProcessorTest : public ::testing::Test {
 protected:
  static const char kTestFeature[];
};

const char GaussianProcessorTest::kTestFeature[] = "test_feature";

TEST_F(GaussianProcessorTest, DoesGetStatsForUpdateHighMultiplicity) {
  // This test verifies that FeatureIndex does job Foo.
  static const int kNumData = 500;
  static const int kNumLevels = 500;
  static const double kInverseVariance = 3.0;
  vector<double> response(kNumData, 1.0);
  vector<double> offsets(kNumData, kInverseVariance);
  vector<double> coefficients(kNumLevels, 1.0);
  vector<double> p_events(kNumData, 2.0);
  vector<double> predicted(kNumLevels);

  // Test without scaling
  {
    static const int kSizeHint = 0;
    MemoryIndexBuilder builder(kTestFeature, kSizeHint);
    for (int i = 0; i < kNumData; ++i) {
      const int seq_index = builder.GetInt64LevelId(i);
      builder.WriteLevelId(seq_index);
    }

    std::unique_ptr<IndexReader> idx = builder.MoveToReader();

    GaussianFeatureProcessor processor;
    // Before calling GetStatsForUpdate, we need to initialize predicted
    // with the immutable part of the sufficient statistic
    for (int i = 0; i < kNumLevels; ++i) {
      predicted[i] = response[i] * offsets[i] * 1.0;  // constant scaling
    }

    processor.GetStatsForUpdate(
        idx.get(), offsets, coefficients, p_events, &predicted);

    for (int i = 0; i < kNumData; ++i) {
      EXPECT_FLOAT_EQ(predicted[i],
        (response[i] - p_events[i] + coefficients[i]) * kInverseVariance);
    }
  }

  // Test with scaling
  {
    vector<double> scaling(kNumData);
    static const int kSizeHint = 0;
    MemoryIndexBuilder builder(kTestFeature, kSizeHint);
    for (int i = 0; i < kNumData; ++i) {
      const int seq_index = builder.GetInt64LevelId(i);
      builder.WriteLevelId(seq_index);
      scaling.push_back((i % 3) + 1);
      builder.MayWriteScaling(scaling[i]);
    }

    std::unique_ptr<IndexReader> idx = builder.MoveToReader();

    GaussianFeatureProcessor processor;
    // Before calling GetStatsForUpdate, we need to initialize predicted
    // with the immutable part of the sufficient statistic
    for (int i = 0; i < kNumLevels; ++i) {
      predicted[i] = response[i] * offsets[i] * scaling[i];  // constant scaling
    }

    processor.GetStatsForUpdate(
        idx.get(), offsets, coefficients, p_events, &predicted);

    for (int i = 0; i < kNumData; ++i) {
      EXPECT_FLOAT_EQ(predicted[i],
                      (response[i] - p_events[i] + coefficients[i])
                      * kInverseVariance * scaling[i]);
    }
  }
}


TEST_F(GaussianProcessorTest, BiasUpdaterDoesGetStatsForUpdate) {
  // This test verifies that BiasIndex correctly implements
  // GetStatsForUpdate
  static const int kNumData = 500;
  static const double kNumDataDbl = static_cast<double>(kNumData);
  static const double kBiasScore = 3.0;
  static const double kResponse = 7.0;
  static const double kPrediction = 3.0;
  static const double kInverseVariance = 3.0;
  vector<double> response(kNumData, kResponse);
  vector<double> offsets(kNumData, kInverseVariance);
  vector<double> coefficients(1, kBiasScore);
  vector<double> p_events(kNumData, 3.0);
  vector<double> predicted(1);

  BiasIndexReader bias_indexer(kNumData);

  GaussianFeatureProcessor processor;
  // Before calling GetStatsForUpdate, we need to initialize predicted
  // with the immutable part of the sufficient statistic
  for (int i = 0; i < kNumData; ++i) {
    predicted[0] += response[i] * offsets[i] * 1.0;  // constant scaling
  }

  processor.GetStatsForUpdate(
      &bias_indexer, offsets, coefficients, p_events, &predicted);
  EXPECT_FLOAT_EQ(predicted[0],
                  (kResponse - kPrediction + kBiasScore)
                  * kNumDataDbl * kInverseVariance);
}

TEST_F(GaussianProcessorTest, DoesAddToPrediction) {
  // This test verifies that FeatureIndex and its subclasses BiasIndex and
  // emre::GaussianUpdater perform AddToPrediction correctly
  static const int kNumData = 4;

  static const double kResponse = 7.0;
  static const double kInverseVariance = 3.0;
  vector<double> response(kNumData, kResponse);
  vector<double> offsets(kNumData, kInverseVariance);

  {
    vector<double> p_events = { 1.0, 1.0, 0.5, 0.1 };
    vector<double> coefficients = { 2.0, 3.0 };
    static const int kSizeHint = 0;
    MemoryIndexBuilder builder(kTestFeature, kSizeHint);
    for (int i = 0; i < kNumData; ++i) {
      int seq_index = i % 2;
      builder.GetInt64LevelId(seq_index);
      builder.WriteLevelId(seq_index);
    }
    std::unique_ptr<IndexReader> idx = builder.MoveToReader();

    vector<double> p_events_new = p_events;

    GaussianFeatureProcessor processor;
    processor.AddToPrediction(idx.get(), coefficients, &p_events_new);
    EXPECT_FLOAT_EQ(p_events_new[0], 2.0 + p_events[0]);
    EXPECT_FLOAT_EQ(p_events_new[1], 3.0 + p_events[1]);
    EXPECT_FLOAT_EQ(p_events_new[2], 2.0 + p_events[2]);
    EXPECT_FLOAT_EQ(p_events_new[3], 3.0 + p_events[3]);
  }

  {
    vector<double> p_events = { 1.0, 1.0, 0.5, 0.1 };
    vector<double> coefficients = { 3.0 };
    vector<double> p_events_new = p_events;
    BiasIndexReader bias_indexer(kNumData);

    GaussianFeatureProcessor processor;
    processor.AddToPrediction(&bias_indexer, coefficients, &p_events_new);
    EXPECT_FLOAT_EQ(p_events_new[0], coefficients[0] + p_events[0]);
    EXPECT_FLOAT_EQ(p_events_new[1], coefficients[0] + p_events[1]);
    EXPECT_FLOAT_EQ(p_events_new[2], coefficients[0] + p_events[2]);
    EXPECT_FLOAT_EQ(p_events_new[3], coefficients[0] + p_events[3]);
  }
}


TEST_F(GaussianProcessorTest, DoesUpdatePredictions) {
  // This test verifies that GaussianUpdater performs UpdatePredictions
  // correctly.
  const int kNumData = 4;

  static const double kResponse = 7.0;
  static const double kInverseVariance = 3.0;
  vector<double> response(kNumData, kResponse);
  vector<double> offsets(kNumData, kInverseVariance);
  vector<double> p_events = { 1.0, 1.0, 0.5, 0.1 };

  {
    MemoryIndexBuilder builder(kTestFeature,
                                           0 /* size_hint */);
    for (int i = 0; i < kNumData; ++i) {
      const int feature_level = i % 2;
      const int seq_index = builder.GetInt64LevelId(feature_level);
      builder.WriteLevelId(seq_index);
    }
    std::unique_ptr<IndexReader> idx = builder.MoveToReader();

    vector<double> coefficient_changes = { 2.0, 0.1 };
    vector<double> p_events_new = p_events;
    GaussianFeatureProcessor processor;
    processor.UpdatePredictions(idx.get(), coefficient_changes, &p_events_new);
    EXPECT_FLOAT_EQ(p_events_new[0], p_events[0] + coefficient_changes[0]);
    EXPECT_FLOAT_EQ(p_events_new[1], p_events[1] + coefficient_changes[1]);
    EXPECT_FLOAT_EQ(p_events_new[2], p_events[2] + coefficient_changes[0]);
    EXPECT_FLOAT_EQ(p_events_new[3], p_events[3] + coefficient_changes[1]);
  }

  {
    BiasIndexReader bias_indexer(kNumData);

    vector<double> coefficient_changes = { -1.3 };
    vector<double> p_events_new = p_events;
    GaussianFeatureProcessor processor;
    processor.UpdatePredictions(&bias_indexer, coefficient_changes,
                                &p_events_new);
    EXPECT_FLOAT_EQ(p_events_new[0], p_events[0] + coefficient_changes[0]);
    EXPECT_FLOAT_EQ(p_events_new[1], p_events[1] + coefficient_changes[0]);
    EXPECT_FLOAT_EQ(p_events_new[2], p_events[2] + coefficient_changes[0]);
    EXPECT_FLOAT_EQ(p_events_new[3], p_events[3] + coefficient_changes[0]);
  }
}


}  // namespace
