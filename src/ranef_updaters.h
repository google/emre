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

// This R library provides a connection to run the essential MCEM steps within
// a single R process itself. The big data structures such as indexers and
// the observation data are stored and accessed on disk.
//  Other essential vector data, such as predicted events per observation
// and coefficients are transparently stored and accessed as R numeric
// vectors without boxed replication.

#ifndef EMRE_RANEF_UPDATERS_H_  // NOLINT
#define EMRE_RANEF_UPDATERS_H_

#include <algorithm>
#include <vector>

#include "indexers/indexer.h"
#include "parameter_updater/update_processor.h"
#include "prior_updater/prior_updater.h"
#include "ranef_updater/block_relaxation.h"
#include "training_data.pb.h"
#include "util/distribution.h"

// We export functions into R with lowercase and underscores.
// The R API has corresponding camelcase functions.

#include <Rcpp.h>

class RanefUpdater {
 public:
  // all components in this structure must be consistent
  struct UpdateTriple {
    std::unique_ptr<emre::PriorUpdater> prior_updater;
    std::unique_ptr<emre::BlockRelaxation> ranef_updater;
    std::unique_ptr<emre::UpdateProcessor> update_processor;
  };

 protected:
  // this constructor is only meant for subclassing
  explicit RanefUpdater(SEXP index_reader_ptr) : indexer_(index_reader_ptr) {}

 public:
  RanefUpdater(const std::string& feature_family_prior_ascii_pb,
               SEXP index_reader_ptr);

  SEXP GetIndexReader() { return(indexer_); }

  // Given coefficients for this feature family, modifies the
  // predicted events per observation p_events_for_obs in-place.
  Rcpp::NumericVector InplaceAddToPrediction(
      Rcpp::NumericVector coefficient_vec,
      Rcpp::NumericVector p_events_for_obs);

  // Collect the mutable part of the sufficient statistics, p_events_for_level,
  // for this feature family. p_events_for_level is modified in-place.
  Rcpp::NumericVector InplaceCollectStats(
      Rcpp::NumericVector offset_vec,
      Rcpp::NumericVector coefficient_vec,
      Rcpp::NumericVector p_events_for_obs,
      Rcpp::NumericVector p_events_for_level);

  // Update coefficient_vec in-place, returns the same vector.
  Rcpp::NumericVector InplaceUpdateRanefCoefficients(
      Rcpp::NumericVector coefficient_vec,
      Rcpp::NumericVector prediction_vec,
      Rcpp::NumericVector ancillary_vec);

  // Update p_events_for_obs in-place, returns the same vector
  Rcpp::NumericVector InplaceUpdatePrediction(
      Rcpp::NumericVector p_events_for_obs,
      Rcpp::NumericVector coefficient_ratio_vec);

  void UpdateRanefPrior(Rcpp::NumericVector coefficient_vec,
                        Rcpp::NumericVector p_events_vec,
                        Rcpp::NumericVector events_vec);

  std::string GetRanefPrior();

 private:
  void SetUpdateMode(const emre::FeatureFamilyPrior& prior);
  void CreateRanefUpdaterTriple(const emre::FeatureFamilyPrior& prior);

 protected:
  void InitializeRandGenerator();

  UpdateTriple updater_;
  emre::FeatureFamilyPrior prior_;
  emre::util::random::Distribution* random_dist_;  // does not own it
  Rcpp::XPtr<emre::IndexReader> indexer_;  // does not own it
};

class ScaledPoissonRanefUpdater : public RanefUpdater {
 public:
  ScaledPoissonRanefUpdater(const std::string& feature_family_prior_ascii_pb,
                            SEXP index_reader_ptr);

 private:
  void SetUpdateMode(const emre::FeatureFamilyPrior& prior);
  void CreateRanefUpdaterTriple(const emre::FeatureFamilyPrior& prior);
};

class GaussianRanefUpdater : public RanefUpdater {
 public:
  GaussianRanefUpdater(const std::string& feature_family_prior_ascii_pb,
                       SEXP index_reader_ptr);

 private:
  void SetUpdateMode(const emre::FeatureFamilyPrior& prior);
  void CreateRanefUpdaterTriple(const emre::FeatureFamilyPrior& prior);
};

#endif  // EMRE_RANEF_UPDATERS_H_  // NOLINT
