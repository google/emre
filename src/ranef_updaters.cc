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

// This Rcpp library provides a connection to run the essential MCEM steps
// within a single R process. All essential data structures such as
// the observation data (events and offsets), predicted events per observation
// and feature level coefficients are transparently stored and accessed as
// R numeric vectors without copies or additional boxing of data structure.

#include "ranef_updaters.h"  // NOLINT

#include <vector>

#include "parameter_updater/gaussian_feature_processor.h"
#include "parameter_updater/poisson_feature_processor.h"
#include "prior_updater/gamma_prior_optimize.h"
#include "prior_updater/gaussian_prior_optimize.h"
#include "ranef_updater/poisson_block_relaxation.h"
#include "ranef_updater/gaussian_block_relaxation.h"
#include "ranef_updater/scaled_poisson_block_relaxation.h"
#include "util/arrayslice.h"
#include "util/distribution.h"
#include "google/protobuf/text_format.h"

// We export functions into R with lowercase and underscores.
// The R API has corresponding camelcase functions.

using Rcpp::CharacterVector;
using Rcpp::NumericVector;
using Rcpp::IntegerVector;

using emre::BlockRelaxation;
using emre::FeatureFamilyPrior;
using emre::IndexReader;
using emre::PriorUpdater;
using emre::UpdateProcessor;

using emre::util::ArraySlice;
using emre::util::MutableArraySlice;

typedef FeatureFamilyPrior FFP;

// TODO(kuehnelf): remove this when update ranef scores is implemented in R
static std::unique_ptr<emre::util::random::Distribution> rng_singleton;

RanefUpdater::RanefUpdater(
    const std::string& feature_family_prior_ascii_pb,
    SEXP index_reader_ptr) : indexer_(index_reader_ptr) {
  // set up update triple, priors and random number generator
  this->InitializeRandGenerator();
  CHECK(google::protobuf::TextFormat::ParseFromString(feature_family_prior_ascii_pb,
                                            &prior_));

  // now set up the ranef updater
  this->CreateRanefUpdaterTriple(prior_);
}

void RanefUpdater::InitializeRandGenerator() {
  // TODO(kuehnelf): remove this when update ranef scores is implemented in R
  // set up random number generator singleton
  if (rng_singleton == nullptr) {
    rng_singleton.reset(new emre::util::random::Distribution(15 /* seed */));
  }
  random_dist_ = rng_singleton.get();
}

NumericVector RanefUpdater::InplaceAddToPrediction(
    NumericVector coefficient_vec,
    NumericVector p_events_for_obs) {
  int num_levels = coefficient_vec.size();
  int num_obs = p_events_for_obs.size();
  CHECK_GT(num_levels, 0);
  CHECK_GT(num_obs, 0);
  CHECK_NOTNULL(indexer_.get());
  CHECK_EQ(indexer_->GetNumLevels(), num_levels);
  CHECK_EQ(indexer_->GetNumObservations(), num_obs);

  // This does not copy the data
  ArraySlice<double> coefficients(coefficient_vec.begin(), num_levels);
  MutableArraySlice<double> p_events(p_events_for_obs.begin(), num_obs);

  updater_.update_processor->AddToPrediction(indexer_, coefficients, p_events);
  return(p_events_for_obs);
}

NumericVector RanefUpdater::InplaceCollectStats(
    NumericVector offset_vec,
    NumericVector coefficient_vec,
    NumericVector p_events_for_obs,
    NumericVector p_events_for_level) {
  CHECK_GT(coefficient_vec.size(), 0);
  CHECK_GE(p_events_for_level.size(), coefficient_vec.size());
  CHECK_GT(p_events_for_obs.size(), 0);
  CHECK_NOTNULL(indexer_.get());
  CHECK_EQ(indexer_->GetNumLevels(), coefficient_vec.size());
  CHECK_EQ(indexer_->GetNumObservations(), p_events_for_obs.size());
  CHECK_EQ(indexer_->GetNumObservations(), offset_vec.size());
  // This does not copy the data
  ArraySlice<double> offset(offset_vec.begin(), offset_vec.size());
  ArraySlice<double> coefficients(coefficient_vec.begin(),
                                  coefficient_vec.size());
  ArraySlice<double> p_events(p_events_for_obs.begin(),
                              p_events_for_obs.size());
  MutableArraySlice<double> prediction(p_events_for_level.begin(),
                                       p_events_for_level.size());

  updater_.update_processor->GetStatsForUpdate(
      indexer_, offset, coefficients, p_events, prediction);
  return(p_events_for_level);
}

// TODO(kuehnelf): this should be implemented in R
// we could get this easily from the update_processor.
NumericVector RanefUpdater::InplaceUpdateRanefCoefficients(
    NumericVector coefficient_vec,
    NumericVector prediction_vec,
    NumericVector ancillary_vec) {
  // only uses the internal prior_ state and the provided arguments
  // to update the ranef scores.
  int num_levels = coefficient_vec.size();
  CHECK_GT(num_levels, 0);
  CHECK_GE(prediction_vec.size(), num_levels);
  CHECK_GE(ancillary_vec.size(), num_levels);

  MutableArraySlice<double> coefficients(coefficient_vec.begin(), num_levels);
  ArraySlice<double> prediction(prediction_vec.begin(), num_levels);
  ArraySlice<double> ancillary(ancillary_vec.begin(), num_levels);

  BlockRelaxation::UpdateParameters params(prediction, ancillary);
  // The prediction are not 'window-smoothed' for updating ranef scores.
  // Gibbs samples new scores, i.e. with the Gamma-Poisson model:
  // draw a random gamma with mean equal to
  //   (prior1 + predicted) divied by
  //   (prior2 + ancillary)
  // i.e. (prior + events) / (prior + expected-events)
  // where expected-events is based on all other features in the model
  updater_.ranef_updater->UpdateRanefs(prior_, params, coefficients);
  return(coefficient_vec);
}

NumericVector RanefUpdater::InplaceUpdatePrediction(
    NumericVector p_events_for_obs,
    NumericVector coefficient_ratio_vec) {
  // no copies here
  ArraySlice<double> coefficient_ratios(coefficient_ratio_vec.begin(),
                                        coefficient_ratio_vec.size());
  MutableArraySlice<double> p_events(p_events_for_obs.begin(),
                                     p_events_for_obs.size());

  updater_.update_processor->UpdatePredictions(
      indexer_, coefficient_ratios, p_events);
  return(p_events_for_obs);
}

void RanefUpdater::UpdateRanefPrior(NumericVector coefficient_vec,
                      NumericVector prediction_vec,
                      NumericVector ancillary_vec) {
  // updates the ranef prior based on the coefficients, the sufficient
  // level stats (for poisson it is p_events as predictions,
  // events as ancillary).
  int num_levels = coefficient_vec.size();
  CHECK_GT(num_levels, 0);
  CHECK_EQ(prediction_vec.size(), num_levels);
  CHECK_EQ(ancillary_vec.size(), num_levels);

  ArraySlice<double> coefficients(coefficient_vec.begin(), num_levels);
  MutableArraySlice<double> prediction(prediction_vec.begin(), num_levels);
  ArraySlice<double> ancillary(ancillary_vec.begin(), num_levels);


  updater_.prior_updater->UpdateVariance(ancillary, prediction,
                                         coefficients, random_dist_);
  // important, update the global state prior consistently
  updater_.prior_updater->SetProtoFromPrior(&prior_);
}

std::string RanefUpdater::GetRanefPrior() {
  FeatureFamilyPrior ffp;
  // this only provides a partial fill of the proto buffer
  updater_.prior_updater->SetProtoFromPrior(&ffp);
  // we augment this information form prior_
  ffp.set_model_class_type(prior_.model_class_type());
  ffp.set_feature_family(prior_.feature_family());
  ffp.set_prior_update_type(prior_.prior_update_type());
  ffp.set_ranef_update_type(prior_.ranef_update_type());

  std::string ffp_str;
  google::protobuf::TextFormat::PrintToString(ffp, &ffp_str);
  return ffp_str;
}

void RanefUpdater::SetUpdateMode(const FeatureFamilyPrior& prior) {
  auto ranef_mode = prior.ranef_update_type();
  auto update_type = prior.prior_update_type();

  // set ranef updater
  if (ranef_mode == FFP::OPTIMIZED) {
    updater_.ranef_updater.reset(new emre::GammaPoissonOptimizer);
  } else {
    updater_.ranef_updater.reset(new emre::GammaPoissonGibbsSampler);
    // TODO(kuehnelf): this API is error prone, easily forgotten to be set up
    updater_.ranef_updater->SetRng(random_dist_);
  }

  // set prior mode
  if (update_type == FFP::SAMPLE) {
    updater_.prior_updater.reset(
        new emre::GammaPoissonPriorSampleOptimize);
  } else if (update_type == FFP::INTEGRATED) {
    updater_.prior_updater.reset(
        new emre::GammaPoissonPriorIntegratedOptimize);
  } else if (update_type == FFP::GIBBS_INTEGRATED) {
    updater_.prior_updater.reset(
        new emre::GammaPoissonPriorIntegratedGibbsSampler);
  } else if (update_type == FFP::DONT_UPDATE) {
    // TODO(kuehnelf): maybe lets make this the default behavior
    updater_.prior_updater.reset(
        new emre::DoesNothingPriorUpdater);
  }
  if (updater_.prior_updater.get() == nullptr) {
    LOG(FATAL) << "prior update type \'"
               << FFP::PriorUpdateType_Name(update_type)
               << "\' not supported";
  }

  updater_.prior_updater->SetPriorFromProto(prior);
}

void RanefUpdater::CreateRanefUpdaterTriple(const FeatureFamilyPrior& prior) {
  CHECK_EQ(prior.model_class_type(), FFP::POISSON);

  updater_.update_processor.reset(new emre::PoissonFeatureProcessor);
  this->SetUpdateMode(prior);
}

// constructors don't call virtual functions, hence we need to implement
// a separate constructor:
ScaledPoissonRanefUpdater::ScaledPoissonRanefUpdater(
    const std::string& feature_family_prior_ascii_pb,
    SEXP index_reader_ptr) : RanefUpdater(index_reader_ptr) {
  // set up update triple, priors and random number generator
  this->InitializeRandGenerator();
  CHECK(google::protobuf::TextFormat::ParseFromString(feature_family_prior_ascii_pb,
                                            &prior_));
  // set up the gaussian ranef updater here
  this->CreateRanefUpdaterTriple(prior_);
}

void ScaledPoissonRanefUpdater::SetUpdateMode(const FeatureFamilyPrior& prior) {
  auto ranef_mode = prior.ranef_update_type();
  auto update_type = prior.prior_update_type();
  // set ranef updater
  if (ranef_mode == FFP::OPTIMIZED) {
    updater_.ranef_updater.reset(new emre::ScaledPoissonOptimizer);
  } else {
    updater_.ranef_updater.reset(new emre::ScaledPoissonGibbsSampler);
    updater_.ranef_updater->SetRng(random_dist_);
  }

  // set prior mode
  if (update_type == FFP::SAMPLE) {
    updater_.prior_updater.reset(new emre::GaussianPriorSampleOptimize);
  } else if (update_type == FFP::DONT_UPDATE) {
    // TODO(kuehnelf): maybe lets make this the default behavior
    updater_.prior_updater.reset(
        new emre::DoesNothingPriorUpdater);
  }
  if (updater_.prior_updater.get() == nullptr) {
    LOG(FATAL) << "prior update type \'"
               << FFP::PriorUpdateType_Name(update_type)
               << "\' not supported";
  }

  updater_.prior_updater->SetPriorFromProto(prior);
}

void ScaledPoissonRanefUpdater::CreateRanefUpdaterTriple(
  const FeatureFamilyPrior& ffp) {
  CHECK_EQ(ffp.model_class_type(), FFP::SCALED_POISSON);

  updater_.update_processor.reset(
      new emre::PoissonScaledFeatureProcessor(indexer_));
  this->SetUpdateMode(ffp);
}

GaussianRanefUpdater::GaussianRanefUpdater(
    const std::string& feature_family_prior_ascii_pb,
    SEXP index_reader_ptr) : RanefUpdater(index_reader_ptr) {
  // set up update triple, priors and random number generator
  this->InitializeRandGenerator();
  CHECK(google::protobuf::TextFormat::ParseFromString(feature_family_prior_ascii_pb,
                                            &prior_));
  // set up the gaussian ranef updater here
  this->CreateRanefUpdaterTriple(prior_);
}

void GaussianRanefUpdater::SetUpdateMode(const FeatureFamilyPrior& prior) {
  auto ranef_mode = prior.ranef_update_type();
  auto update_type = prior.prior_update_type();
  // set ranef mode
  if (ranef_mode == FFP::OPTIMIZED) {
    updater_.ranef_updater.reset(new emre::gaussian::GaussianOptimizer);
  } else {
    updater_.ranef_updater.reset(
        new emre::gaussian::GaussianGibbsSampler);
    updater_.ranef_updater->SetRng(random_dist_);
  }

  // set prior mode
  if (update_type == FFP::SAMPLE) {
    updater_.prior_updater.reset(new emre::GaussianPriorSampleOptimize);
  } else if (update_type == FFP::INTEGRATED) {
    updater_.prior_updater.reset(
        new emre::GaussianPriorIntegratedOptimize);
  } else if (update_type == FFP::GIBBS_INTEGRATED) {
    updater_.prior_updater.reset(new emre::GaussianPriorGibbsSampler);
  } else if (update_type == FFP::DONT_UPDATE) {
    // TODO(kuehnelf): maybe lets make this the default behavior
    updater_.prior_updater.reset(
        new emre::DoesNothingPriorUpdater);
  }
  if (updater_.prior_updater.get() == nullptr) {
    LOG(FATAL) << "prior update type \'"
               << FFP::PriorUpdateType_Name(update_type)
               << "\' not supported";
  }

  updater_.prior_updater->SetPriorFromProto(prior);
}

void GaussianRanefUpdater::CreateRanefUpdaterTriple(
  const FeatureFamilyPrior& ffp) {
  CHECK_EQ(ffp.model_class_type(), FFP::GAUSSIAN);

  updater_.update_processor.reset(new emre::GaussianFeatureProcessor);
  this->SetUpdateMode(ffp);
}

RCPP_MODULE(mod_ranef_updater) {
  using Rcpp::class_;

  // TODO(kuehnelf): investigate to use
  // .field("coefficients", &RanefUpdater::coefficient)
  class_<RanefUpdater>("RanefUpdater")
  .constructor<std::string, SEXP>()
  .method("get.index.reader", &RanefUpdater::GetIndexReader)
  .method("inp.addto.prediction", &RanefUpdater::InplaceAddToPrediction)
  .method("inp.collect.stats", &RanefUpdater::InplaceCollectStats)
  .method("inp.update.prediction",
          &RanefUpdater::InplaceUpdatePrediction)
  .method("inp.update.ranef.coefficients",
          &RanefUpdater::InplaceUpdateRanefCoefficients)
  .method("update.ranef.prior", &RanefUpdater::UpdateRanefPrior)
  .method("get.ranef.prior", &RanefUpdater::GetRanefPrior);

  class_<ScaledPoissonRanefUpdater>("ScaledPoissonRanefUpdater")
  .derives<RanefUpdater>("RanefUpdater")
  .constructor<std::string, SEXP>();

  class_<GaussianRanefUpdater>("GaussianRanefUpdater")
  .derives<RanefUpdater>("RanefUpdater")
  .constructor<std::string, SEXP>();
}
