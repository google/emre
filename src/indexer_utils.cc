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

// We export functions into R with lowercase and underscores.
// The R API has corresponding camelcase functions.

#include "base/logging.h"
#include "base/stringprintf.h"
#include "indexers/bias_indexer.h"
#include "indexers/memory_indexer.h"

// Rcpp defines an ERROR macro that interferes with base/logging.h
#include <Rcpp.h>

namespace {

const char kNAFeature[] = "NA_feature";
const double kDefaultScaling = 1.0;

using emre::BiasIndexReader;
using emre::IndexBuilder;
using emre::IndexReader;
using emre::MemoryIndexBuilder;
using emre::MemoryIndexReader;
using emre::FeatureData;

using Rcpp::CharacterVector;
using Rcpp::DoubleVector;
using Rcpp::IntegerVector;

class FeatureIndexMemoryWriter {
 public:
  explicit FeatureIndexMemoryWriter(const std::string& feature_family)
      : feature_family_(feature_family),
        index_builder_(new MemoryIndexBuilder(feature_family)) {}

  int WriteStringLevelFeatures(CharacterVector features) {
    FeatureData datum;
    const int num_elem = features.size();
    CharacterVector::iterator feat_iter = features.begin();

    for (int i = 0; i < num_elem; ++i, ++feat_iter) {
      if (CharacterVector::is_na(*feat_iter)) {
        datum.SetFeature(kNAFeature, "");
      } else {
        std::string level(*feat_iter);
        datum.SetFeature(feature_family_, level);
      }
      index_builder_->ProcessData(datum);
    }

    return num_elem;
  }

  int WriteStringLevelFeaturesWithScaling(CharacterVector features,
                                          DoubleVector scaling) {
    FeatureData datum;
    const int num_elem = features.size();
    CharacterVector::iterator feat_iter = features.begin();
    DoubleVector::iterator scal_iter = scaling.begin();

    for (int i = 0; i < num_elem; ++i, ++feat_iter, ++scal_iter) {
      if (CharacterVector::is_na(*feat_iter)) {
        datum.SetFeatureWithScaling(kNAFeature, "", kDefaultScaling);
      } else {
        std::string level(*feat_iter);
        datum.SetFeatureWithScaling(feature_family_, level, *scal_iter);
      }
      index_builder_->ProcessData(datum);
    }

    return num_elem;
  }

  int WriteInt64LevelFeatures(IntegerVector features) {
    FeatureData datum;
    const int num_elem = features.size();
    IntegerVector::iterator feat_iter = features.begin();

    for (int i = 0; i < num_elem; ++i, ++feat_iter) {
      if (IntegerVector::is_na(*feat_iter)) {
        datum.SetFeatureInt64(kNAFeature, -1);
      } else {
        int64 feature_id = *feat_iter;
        datum.SetFeatureInt64(feature_family_, feature_id);
      }
      index_builder_->ProcessData(datum);
    }

    return num_elem;
  }

  SEXP Close() {
    std::unique_ptr<IndexReader> index = index_builder_->MoveToReader();
    index_builder_.reset(nullptr);
    CHECK_NOTNULL(index.get());
    // register a finalizer for R when index_reader_ptr is garbage collected
    Rcpp::XPtr<IndexReader> index_reader_ptr(index.release());
    // the class attribute can be resovled in R with class(object)
    index_reader_ptr.attr("class") = Rcpp::wrap("IndexReader");

    return index_reader_ptr;
  }

 private:
  const std::string feature_family_;
  std::unique_ptr<IndexBuilder> index_builder_;
};

// Function must return an SEXP
RcppExport SEXP get_string_levels(SEXP index_reader_ptr) {
  Rcpp::XPtr<IndexReader> indexer(index_reader_ptr);

  BEGIN_RCPP;
  const int num_levels = indexer->GetNumLevels();
  const std::string family_name = indexer->GetFeatureFamily();
  CharacterVector r(num_levels);
  FeatureData feature_data;

  // we need to reset the sequential VectorReader for actual use.
  indexer->ResetFeatureLevel();
  for (int i = 0; i < num_levels; ++i) {
    feature_data.UnsetAll();
    indexer->FillFeatureLevel(&feature_data);
    CHECK(feature_data.HasFeature(family_name));

    std::string feature_str;
    if (feature_data.HasStringValues(family_name)) {
      feature_str = feature_data.FeatureValue(family_name);
    } else {
      emre::SStringPrintf(&feature_str, "%7" GG_LL_FORMAT "d",
                          feature_data.FeatureValueInt64(family_name));
    }
    r[i] = feature_str;
  }
  return r;
  END_RCPP;
}

// Function must return an SEXP
RcppExport SEXP get_levelid_map(SEXP index_reader_ptr) {
  Rcpp::XPtr<IndexReader> indexer(index_reader_ptr);

  BEGIN_RCPP;
  int n_obs = indexer->GetNumObservations();
  IntegerVector r(n_obs);

  auto level_iter = indexer->GetLevelIterator();
  for (int i = 0; i < n_obs && !level_iter.Done(); ++i) {
    r[i] = level_iter.Next();
  }

  return r;
  END_RCPP;
}

// Function must return an SEXP
RcppExport SEXP get_row_scaling(SEXP index_reader_ptr) {
  Rcpp::XPtr<IndexReader> indexer(index_reader_ptr);

  BEGIN_RCPP;
  int n_obs = indexer->GetNumObservations();
  DoubleVector r(n_obs);

  auto scaling_iter = indexer->GetScalingIterator();
  for (int i = 0; i < n_obs && !scaling_iter.Done(); ++i) {
    r[i] = scaling_iter.Next();
  }

  return r;
  END_RCPP;
}

// Function must return an SEXP
RcppExport SEXP get_num_levels(SEXP index_reader_ptr) {
  Rcpp::XPtr<IndexReader> indexer(index_reader_ptr);

  BEGIN_RCPP;
  IntegerVector r(1);
  r[0] = indexer->GetNumLevels();
  return r;
  END_RCPP;
}

// Function must return an SEXP
RcppExport SEXP get_num_observations(SEXP index_reader_ptr) {
  Rcpp::XPtr<IndexReader> indexer(index_reader_ptr);

  BEGIN_RCPP;
  IntegerVector r(1);
  r[0] = indexer->GetNumObservations();
  return r;
  END_RCPP;
}

RcppExport SEXP create_bias_index_reader(SEXP num_observations) {
  int n_obs = Rcpp::as<int>(num_observations);

  Rcpp::XPtr<IndexReader> index_reader_ptr(
      new BiasIndexReader(n_obs), true);
  // the class attribute can be resovled in R with class(object)
  index_reader_ptr.attr("class") = Rcpp::wrap(std::vector<std::string>
      {"IndexReader", "BiasIndexReader"});
  return index_reader_ptr;
}

}  // end namespace

RCPP_MODULE(mod_indexer_utils) {
  using Rcpp::class_;

  class_<FeatureIndexMemoryWriter>("FeatureIndexMemoryWriter")

  .constructor<std::string>()
  .method("write.string.features",
          &FeatureIndexMemoryWriter::WriteStringLevelFeatures)
  .method("write.scaled.string.features",
          &FeatureIndexMemoryWriter::WriteStringLevelFeaturesWithScaling)
  .method("write.int64.features",
          &FeatureIndexMemoryWriter::WriteInt64LevelFeatures)
  // we don't need a finalizer here because the ObservationWriter
  // class destructor takes care of everything here
  .method("close", &FeatureIndexMemoryWriter::Close);
}
