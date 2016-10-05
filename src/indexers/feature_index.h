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

#ifndef EMRE_INDEXERS_FEATURE_INDEX_H_  // NOLINT
#define EMRE_INDEXERS_FEATURE_INDEX_H_

#include <algorithm>
#include <cassert>
#include <string>
#include <unordered_map>

#include "base/integral_types.h"

namespace emre {

// This class is a temporary way to store the features active in a given
// observation.  The family/level pairs it
class FeatureData {
 public:
  ~FeatureData() {}

  bool HasFeature(const std::string& feature_name) const {
    return (int64_features_.find(feature_name) != int64_features_.end())
        || (string_features_.find(feature_name) != string_features_.end());
  }

  bool HasStringValues(const std::string& feature_name) const {
    return (string_features_.find(feature_name) != string_features_.end());
  }

  void UnsetFeature(const std::string& feature_name) {
    if (!HasFeature(feature_name)) {
      return;
    } else if (HasStringValues(feature_name)) {
      string_features_.erase(feature_name);
    } else {
      int64_features_.erase(feature_name);
    }
  }

  void UnsetAll() {
    string_features_.clear();
    int64_features_.clear();
    scaling_.clear();
  }

  const std::string& FeatureValue(const std::string& feature_name) const {
    auto it = string_features_.find(feature_name);
    assert(it != string_features_.end());
    return it->second;
  }

  int64 FeatureValueInt64(const std::string& feature_name) const {
    auto it = int64_features_.find(feature_name);
    assert(it != int64_features_.end());
    return it->second;
  }

  double FeatureScaling(const std::string& feature_name,
                        double default_value) const {
    auto it = scaling_.find(feature_name);
    if (it == scaling_.end()) {
      return default_value;
    }
    return it->second;
  }

  void SetFeature(const std::string& feature_name,
                  const std::string& feature_value) {
    string_features_[feature_name] = feature_value;
  }

  void SetFeatureWithScaling(const std::string& feature_name,
                             const std::string& feature_value,
                             double scaling) {
    string_features_[feature_name] = feature_value;
    scaling_[feature_name] = scaling;
  }

  void SetFeatureInt64(const std::string& feature_name,
                       int64 feature_value) {
    int64_features_[feature_name] = feature_value;
  }

 private:
  std::unordered_map<std::string, int64> int64_features_;
  std::unordered_map<std::string, std::string> string_features_;
  std::unordered_map<std::string, double> scaling_;
};

}  // namespace emre

#endif  // EMRE_INDEXERS_FEATURE_INDEX_H_  // NOLINT
