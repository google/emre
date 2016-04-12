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

#include "vector_storage.h"  // NOLINT

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

using testing::Return;

namespace emre {

template <typename T>
class MockVectorReader : public VectorReader<T> {
 public:
  MOCK_CONST_METHOD0_T(Size, int());
  MOCK_CONST_METHOD0_T(AsVector, vector<T>*());
  MOCK_METHOD0_T(Reset, void());
  MOCK_METHOD0_T(Next, T());
};

TEST(VectorStorage, IteratorWrapsNext) {
  MockVectorReader<int> vec_reader;
  EXPECT_CALL(vec_reader, Size())
      .Times(1)
      .WillOnce(Return(3));
  EXPECT_CALL(vec_reader, Reset())
      .Times(1);
  EXPECT_CALL(vec_reader, AsVector())
      .Times(1)
      .WillOnce(Return(nullptr));
  EXPECT_CALL(vec_reader, Next())
      .Times(3)
      .WillOnce(Return(1))
      .WillOnce(Return(2))
      .WillOnce(Return(3));

  vector<int> results;
  for (auto itr = vec_reader.GetIterator(); !itr.Done();) {
    results.push_back(itr.Next());
  }

  vector<int> expected = {1, 2, 3};
  EXPECT_EQ(expected, results);
}

TEST(VectorStorage, IteratorWrapsAsVector) {
  MockVectorReader<int> vec_reader;
  EXPECT_CALL(vec_reader, Size())
      .Times(1)
      .WillOnce(Return(3));
  EXPECT_CALL(vec_reader, Reset())
      .Times(1);
  vector<int> expected = {4, 5, 6};
  EXPECT_CALL(vec_reader, AsVector())
      .Times(1)
      .WillOnce(Return(&expected));
  EXPECT_CALL(vec_reader, Next())
      .Times(0);

  vector<int> results;
  for (auto itr = vec_reader.GetIterator(); !itr.Done();) {
    results.push_back(itr.Next());
  }

  EXPECT_EQ(expected, results);
}

TEST(ConstVectorReader, Test) {
  ConstVectorReader<int> const_vec_reader(2014, 100);
  EXPECT_EQ(100, const_vec_reader.Size());
  auto itr = const_vec_reader.GetIterator();
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(2014, itr.Next());
  }
}

}  // namespace emre
