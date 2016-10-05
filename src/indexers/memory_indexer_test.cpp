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

#include "memory_indexer.h"  // NOLINT
#include "gtest/gtest.h"

namespace {

typedef emre::FeatureData FeatureData;
typedef emre::IndexReader IndexReader;
typedef emre::MemoryIndexBuilder MemoryIndexBuilder;

class MemoryIndexerTest : public ::testing::Test {
 protected:
  static const char kTestFeature[];
  static const char kOtherFeature[];
};

const char MemoryIndexerTest::kTestFeature[] = "test_feature";
const char MemoryIndexerTest::kOtherFeature[] = "other_feature";

TEST_F(MemoryIndexerTest, HandleMissingFeature) {
  MemoryIndexBuilder idx(kTestFeature, 3);
  FeatureData datum1, datum2, datum3;
  // 3 observations with 2 levels for test feature and missing level (-1)
  datum1.SetFeature(kTestFeature, "a");
  datum2.SetFeature(kOtherFeature, "0");
  datum3.SetFeature(kTestFeature, "b");
  idx.ProcessData(datum1);
  idx.ProcessData(datum2);
  idx.ProcessData(datum3);
  EXPECT_EQ(idx.NextLevelId(), 2);

  std::unique_ptr<IndexReader> index = idx.MoveToReader();
  EXPECT_EQ(index->GetNumLevels(), 2);
  EXPECT_EQ(index->GetNumObservations(), 3);

  auto iter = index->GetLevelIterator();
  EXPECT_EQ(iter.Next(), 0);
  EXPECT_EQ(iter.Next(), -1);  // observation has this feature missing!
  EXPECT_EQ(iter.Next(), 1);
  EXPECT_TRUE(iter.Done());
}

TEST_F(MemoryIndexerTest, BuildMemoryTinyIndexAndRead) {
  static const int kNumData = 5;
  {
    MemoryIndexBuilder idx(kTestFeature, 3);
    for (int i = 0; i < kNumData; ++i) {
      if (i % 2 == 0) {
        int val = idx.GetInt64LevelId(i % 2);
        idx.WriteLevelId(val);
      } else {
        // missing feature
        idx.WriteLevelId(-1);
      }
    }
    EXPECT_EQ(idx.NextLevelId(), 1);

    std::unique_ptr<IndexReader> index = idx.MoveToReader();
    // bool index is somewhat special: A missing feature is a zero bit,
    // while the bit is set to one for an existing feature.
    EXPECT_EQ(index->GetNumLevels(), 1);
    auto iter = index->GetLevelIterator();
    for (int i = 0; i < kNumData && !iter.Done(); i++) {
      EXPECT_EQ(iter.Next(), -(i % 2));
    }
  }
}

TEST_F(MemoryIndexerTest, BuildMemorySmallIndexAndRead) {
  static const int kNumData = 255;
  static const int kNumLevels = 255;  // must fit in uint8
  {
    MemoryIndexBuilder idx(kTestFeature);
    for (int i = 0; i < kNumData; ++i) {
      int val = idx.GetInt64LevelId(i);
      idx.WriteLevelId(val);
    }
    EXPECT_EQ(idx.NextLevelId(), kNumData);

    std::unique_ptr<IndexReader> index = idx.MoveToReader();
    EXPECT_EQ(index->GetNumLevels(), kNumLevels);
    auto iter = index->GetLevelIterator();
    for (int i = 0; i < kNumData && !iter.Done(); i++) {
      EXPECT_EQ(iter.Next(), i);
    }
  }
}

TEST_F(MemoryIndexerTest, BuildMemoryMediumIndexWithScalingAndRead) {
  static const int kNumData = 65535;
  static const int kNumLevels = 65535;  // must fit in vector<uint16>
  {
    MemoryIndexBuilder idx(kTestFeature);
    for (int i = 0; i < kNumData; ++i) {
      int val = idx.GetInt64LevelId(i + 3);
      idx.WriteLevelId(val);
      idx.MayWriteScaling(3.5 * sqrt(static_cast<double>(i)));
    }
    EXPECT_EQ(idx.NextLevelId(), kNumData);

    std::unique_ptr<IndexReader> index = idx.MoveToReader();
    EXPECT_EQ(index->GetNumLevels(), kNumLevels);
    auto level_iter = index->GetLevelIterator();
    auto scaling_iter = index->GetScalingIterator();
    for (int i = 0; i < kNumData && !level_iter.Done() && !scaling_iter.Done();
         ++i) {
      EXPECT_EQ(level_iter.Next(), i);
      EXPECT_FLOAT_EQ(scaling_iter.Next(), 3.5 * sqrt(static_cast<double>(i)));
    }
  }
}

TEST_F(MemoryIndexerTest, BuildMemoryLargeIndexAndRead) {
  static const int kNumData = 65536;
  static const int kNumLevels = 65536;  // this fits in vector<int32>
  {
    MemoryIndexBuilder idx(kTestFeature, kNumLevels);
    for (int i = 0; i < kNumData; ++i) {
      int val = idx.GetInt64LevelId(i + 7);
      idx.WriteLevelId(val);
    }
    EXPECT_EQ(idx.NextLevelId(), kNumData);

    std::unique_ptr<IndexReader> index = idx.MoveToReader();
    EXPECT_EQ(index->GetNumLevels(), kNumLevels);
    auto iter = index->GetLevelIterator();
    for (int i = 0; i < kNumData; i++) {
      EXPECT_EQ(iter.Next(), i);
    }
  }
}

}  // namespace
