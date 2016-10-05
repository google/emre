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

// Utility functions for updating scaled or "continuous" features in the
// Poisson model.  Unlike unscaled features these will have a log-normal prior
// rather than a Gamma since there is no computational benefit to using a Gamma
// prior for this type of feature.
//


#ifndef EMRE_PARAMETER_UPDATER_SCALED_FEATURE_UTIL_H_  // NOLINT
#define EMRE_PARAMETER_UPDATER_SCALED_FEATURE_UTIL_H_

#include <vector>
#include "indexers/indexer.h"
#include "indexers/vector_storage.h"
#include "util/arrayslice.h"
#include "util/basic_types.h"
#include "util/distribution.h"
#include "prior_updater/metropolis_hastings.h"

namespace emre {

class ScaledFeatureUtil {
 public:
  // For each chunk of data we have a 'level' and 'scaling' i.e. the feature
  // level and a continuous value for that feature.  We want to aggregate
  // to the unique values of (level, scaling) so that the MCMC updates can
  // be performed quickly.
  //
  // The pairs (level, scaling) for which level = k map to range
  //   posn, posn + 1, ..., posn + size - 1
  // where posn = level_posn_size[k].first and
  // size = level_posn_size[k].second
  //
  // The pair (level[i], scaling[i]) maps to say position 'm' and we will have
  //   posn <= m < posn + size
  // where again (posn, size) = level_posn_size[level[i]].  We will also
  // aggregate_scaling[m] = scaling[i]
  static void MakeLevelScalingMapping(
      IndexReader* index,
      std::vector<std::pair<int, int>>* level_posn_size,
      VectorBuilder<int>* level_scaling_posn,
      std::vector<double>* aggregate_scaling);

  // Assumes that 'prediction' is already allocated and zero'ed out
  // 'scores' are in log-scale i.e. pconvs = exp(score) * offset
  static void GetPredictionForPoissonUpdate(
      IndexReader* index,
      VectorReader<int>* level_scaling_posn,
      util::ArraySlice<double> scores,
      util::ArraySlice<double> p_events,
      util::MutableArraySlice<double> prediction);

  // Assumes that 'events' is already allocated and zero'ed out
  // Computes the total events per aggregation level
  static void GetEventsForPoissonUpdate(
      IndexReader* index,
      VectorReader<int>* level_scaling_posn,
      util::ArraySlice<double> events,
      util::MutableArraySlice<double> events_per_agg_level);

  // Computes a likelihood for each random effect assuming it takes the value
  // in 'scores'.  The vector of returned 'likelihoods' is assumed to be
  // already zero'ed out and of proper size.
  static void GetDataLikelihoodsPoissonUpdate(
      IndexReader* index,
      util::ArraySlice<double> events,
      util::ArraySlice<double> scores,
      util::ArraySlice<double> p_events,
      util::MutableArraySlice<double> likelihoods);

  // p_events should not include the contribution from 'scores'
  static void GaussianProposalMHPoissonUpdate(
      int num_steps, double prior_inverse_variance,
      IndexReader* index,
      util::ArraySlice<double> events,
      util::ArraySlice<double> p_events,
      util::ArraySlice<double> scores,
      util::ArraySlice<double> proposal_sds,
      std::vector<double>* final_scores,
      util::MutableArraySlice<int> num_accept,
      util::random::Distribution* distn);
};

namespace poisson {
// Does not own 'distn'
class ScaledFeatureMhProposer
    : public emre::metropolis_hastings::MhProposer<double> {
  // Currently, this proposer doesn't use any of the predicted events, events
  // nor aggregate scaling data.
 public:
  ScaledFeatureMhProposer(
      util::ArraySlice<std::pair<int, int>> level_posn_size,
      util::ArraySlice<double> aggregate_scaling,
      util::ArraySlice<double> prediction,
      util::ArraySlice<double> events,
      util::ArraySlice<double> proposal_sds,
      util::random::Distribution* distn);

  virtual ~ScaledFeatureMhProposer() {}

  int GetCurrentLevelIndex() const { return current_level_index_; }
  inline void IncrementLevelIndex() {
    // CHECK_LT(current_level_index_, level_posn_size_.size());
    ++current_level_index_;
  }

  void GenerateProposal(const double& from, double* to,
                        double* proposal_llik) override {
    double current_sd = proposal_sds_[current_level_index_];
    if (proposal_llik != nullptr) *proposal_llik = 0.0;
    *to = from + distn_->RandGaussian(current_sd);
  }

  // This is only used to compute the Metropolis-Hastings ratio and since
  // the proposal is symmetric, we can simply return 0.0
  double ProposalLoglikForMhRatio(const double& from,
                                  const double& to) override {
    return 0.0;
  }

 private:
  util::ArraySlice<std::pair<int, int>> level_posn_size_;
  util::ArraySlice<double> aggregate_scaling_;
  util::ArraySlice<double> prediction_;
  util::ArraySlice<double> events_;
  util::ArraySlice<double> proposal_sds_;
  util::random::Distribution* distn_;

  int current_level_index_;
};

class ScaledFeatureLoglik
    : public emre::LogLikelihodFunction<double> {
 public:
  virtual ~ScaledFeatureLoglik() {}

  ScaledFeatureLoglik(
      double inverse_variance,
      util::ArraySlice<std::pair<int, int>> level_posn_size,
      util::ArraySlice<double> aggregate_scaling,
      util::ArraySlice<double> prediction,
      util::ArraySlice<double> events);

  int GetCurrentLevelIndex() const { return current_level_index_; }
  void IncrementLevelIndex();

  // Assumes that 'r' is non-NULL and has length at least that of 'x'
  void Evaluate(util::ArraySlice<double> x,
                util::MutableArraySlice<double> r) const override;

  // computes the derivative with respect to x
  double EvaluateDx(const double x) const;

 private:
  const double inverse_variance_;
  // first int points to index of a unique level in the flattened
  // unique level-scaling vector, second int is the number of unique scalings
  // for its level.
  util::ArraySlice<std::pair<int, int>> level_posn_size_;
  // array of scaling for each unique level-scaling
  util::ArraySlice<double> aggregate_scaling_;

  util::ArraySlice<double> prediction_;
  util::ArraySlice<double> events_;

  int current_level_index_;  // index into level_posn_size_
  util::ArraySlice<double> current_scaling_;
  util::ArraySlice<double> current_prediction_;
  util::ArraySlice<double> current_events_;
};

class ScaledFeatureRootSolver {
 public:
  explicit ScaledFeatureRootSolver(
      ScaledFeatureLoglik* llik,
      const int max_find_bounds_iter = 15,
      const int max_brent_iter = 15,
      const double solution_precision = 0.01)
      : llik_(llik),
        max_bound_iter_(max_find_bounds_iter),
        max_brent_iter_(max_brent_iter),
        epsilon_(solution_precision) {}

  int Solve(double* root);

 private:
  void InitializeBracket(double start);

 private:
  ScaledFeatureLoglik* llik_;
  const int max_bound_iter_;  // 2^max_bound_iter_ * 0.05 to find bounds
  const int max_brent_iter_;  // maximum bi-section/brent iterations
  const double epsilon_;  //  accuracy of the solution
  double lb_;  // lower bound for solution
  double ub_;  // upper bound for solution
};

}  // namespace poisson
}  // namespace emre

#endif  // EMRE_PARAMETER_UPDATER_SCALED_FEATURE_UTIL_H_  // NOLINT
