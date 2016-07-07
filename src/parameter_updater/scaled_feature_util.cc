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

#include "scaled_feature_util.h"  // NOLINT

#include <algorithm>
#include <map>

#include "gsl/gsl_errno.h"
#include "gsl/gsl_roots.h"
#include "base/logging.h"
#include "indexers/vector_storage.h"

namespace emre {

using util::ArraySlice;
using util::MutableArraySlice;

// static
void ScaledFeatureUtil::MakeLevelScalingMapping(
    IndexReader* index,
    vector<pair<int, int>>* level_posn_size,
    VectorBuilder<int>* level_scaling_posn,
    vector<double>* aggregate_scaling) {

  // First count the unique (level, scaling) pairs
  int max_level = -1;
  std::map<int, int> num_scaling_for_level;
  std::map<pair<int, double>, int> unique_level_scaling;

  auto level_iter = index->GetLevelIterator();
  auto scaling_iter = index->GetScalingIterator();

  while (!level_iter.Done()) {
    const int lev = level_iter.Next();
    const double scaling = scaling_iter.Next();
    if (lev >= 0) {
      max_level = std::max(max_level, lev);

      int scaling_index = 0;
      auto it = num_scaling_for_level.find(lev);
      if (it != num_scaling_for_level.end()) {
        scaling_index = it->second;
      }
      auto lev_scaling = pair<int, double>(lev, scaling);
      if (unique_level_scaling.insert(pair<pair<int, double>, int>(
          lev_scaling, scaling_index)).second) {
        auto ret = num_scaling_for_level.insert(
            pair<int, int>(lev, scaling_index + 1));
        if (!ret.second) {
          ret.first->second = scaling_index + 1;
        }
      }
    }
  }

  // Compute positions for each level in flattened vector.  The positions will
  // be spaced by the number of unique 'scaling' values we found for that
  // level.
  level_posn_size->clear();
  level_posn_size->reserve(max_level);
  int current_position = 0;
  for (int i = 0; i <= max_level; ++i) {
    int current_size = 0;
    auto it = num_scaling_for_level.find(i);
    if (it != num_scaling_for_level.end()) {
      current_size = it->second;
    }
    level_posn_size->emplace_back
        (current_position, current_size);
    current_position += current_size;
  }

  // In the flattened vector record the value of the scaling in that position.
  aggregate_scaling->assign(current_position, 0);
  for (const auto& x : unique_level_scaling) {
    const int level_index = x.first.first;
    const int scaling_index = x.second;
    // CHECK_NE(level_index, -1);
    // CHECK_LT(level_index, level_posn_size->size());
    const int posn = (*level_posn_size)[level_index].first + scaling_index;
    // CHECK_LT(posn, aggregate_scaling->size());
    CHECK_GE(posn, 0);
    (*aggregate_scaling)[posn] = x.first.second;
  }

  // Finally save a map from the indices into the inputs 'level' and 'scaling'
  // into the flattened vector.  This mapping will be used to aggregate
  // statistics needed for sampling.
  level_iter.Reset();
  scaling_iter.Reset();
  while (!level_iter.Done()) {
    const int lev = level_iter.Next();
    const double scaling = scaling_iter.Next();
    if (lev < 0) {
      level_scaling_posn->Write(-1);
      continue;
    }
    const pair<int, double> level_scaling(lev, scaling);
    const int posn = (*level_posn_size)[lev].first
                     + unique_level_scaling[level_scaling];
    level_scaling_posn->Write(posn);
  }
}

// static
void ScaledFeatureUtil::GetPredictionForPoissonUpdate(
    IndexReader* index,
    VectorReader<int>* level_scaling_posn,
    ArraySlice<double> coefficients,
    ArraySlice<double> p_events,
    MutableArraySlice<double> prediction) {
  const int n_obs = index->GetNumObservations();
  CHECK_EQ(n_obs, level_scaling_posn->Size());
  auto p_events_iter = p_events.begin();
  auto level_iter = index->GetLevelIterator();
  auto posn_iter = level_scaling_posn->GetIterator();
  auto scaling_iter = index->GetScalingIterator();
  for (int i = 0; i < n_obs && !level_iter.Done(); ++i, ++p_events_iter) {
    const int aggregation_index = posn_iter.Next();
    const int level_index = level_iter.Next();
    const double scaling = scaling_iter.Next();

    if (level_index >= 0) {
      CHECK_GE(aggregation_index, 0);
      CHECK_LT(aggregation_index, prediction.size());
      CHECK_LT(level_index, coefficients.size());

      // To remove the feature's contribution to 'p_events' we must also
      // multiply in the scaling
      prediction[aggregation_index] +=
          (*p_events_iter) * exp(-scaling * coefficients[level_index]);
    }
  }
}

void ScaledFeatureUtil::GetEventsForPoissonUpdate(
    IndexReader* index,
    VectorReader<int>* level_scaling_posn,
    ArraySlice<double> events,
    MutableArraySlice<double> events_stats) {
  const int n_obs = events.size();
  CHECK_EQ(n_obs, level_scaling_posn->Size());

  auto level_iter = index->GetLevelIterator();
  auto posn_iter = level_scaling_posn->GetIterator();
  auto events_iter = events.begin();
  for (int i = 0; i < n_obs && !level_iter.Done(); ++i, ++events_iter) {
    const int aggregation_index = posn_iter.Next();
    const int level_index = level_iter.Next();
    if (level_index >= 0) {
      CHECK_GE(aggregation_index, 0);
      CHECK_LT(aggregation_index, events_stats.size());
      events_stats[aggregation_index] += *events_iter;
    }
  }
}

// static
void ScaledFeatureUtil::GetDataLikelihoodsPoissonUpdate(
    IndexReader* index,
    ArraySlice<double> events,
    ArraySlice<double> coefficients,
    ArraySlice<double> p_events,
    MutableArraySlice<double> likelihoods) {
  // It is assumed that likelihood has the proper size
  auto level_iter = index->GetLevelIterator();
  auto scaling_iter = index->GetScalingIterator();
  auto p_events_iter = p_events.begin();
  auto events_iter = events.begin();
  for (int i = 0; !level_iter.Done(); ++i, ++events_iter, ++p_events_iter) {
    const int level_index = level_iter.Next();
    const double scl = scaling_iter.Next();
    if (level_index >= 0) {
      const double scl_coeff = scl * coefficients[level_index];
      likelihoods[level_index] += -(*p_events_iter) * exp(scl_coeff)
                                  + (*events_iter) * scl_coeff;
    }
  }
}

// static
void ScaledFeatureUtil::GaussianProposalMHPoissonUpdate(
    int num_steps, double prior_inverse_variance,
    IndexReader* index,
    ArraySlice<double> events,
    ArraySlice<double> p_events,
    ArraySlice<double> scores,
    ArraySlice<double> proposal_sds,
    vector<double>* final_scores,
    MutableArraySlice<int> num_accept,
    util::random::Distribution* distn) {
  vector<double> likelihoods(scores.size(), 0.0);
  for (int i = 0; i < scores.size(); ++i) {
    likelihoods[i] = -0.5 * scores[i] * scores[i] * prior_inverse_variance;
  }
  ScaledFeatureUtil::GetDataLikelihoodsPoissonUpdate(
      index, events, scores, p_events, &likelihoods);

  vector<double> current_scores(scores.size());
  std::copy(scores.begin(), scores.end(), current_scores.begin());
  vector<double> proposal_likelihoods(scores.size(), 0.0);
  vector<double> proposal_scores(scores.size(), 0.0);

  for (int iter = 0; iter < num_steps; ++iter) {
    // Make the proposal and get the prior likelihood
    for (int i = 0; i < scores.size(); ++i) {
      proposal_scores[i] = current_scores[i]
                           + distn->RandGaussian(proposal_sds[i]);
      proposal_likelihoods[i] =
          -0.5 * proposal_scores[i] * proposal_scores[i]
          * prior_inverse_variance;
    }

    // Add in the data likelihood
    ScaledFeatureUtil::GetDataLikelihoodsPoissonUpdate(
        index, events, proposal_scores, p_events, &proposal_likelihoods);

    // Compute the Metropolis-Hastings ratios and accept/reject each proposal
    for (int i = 0; i < scores.size(); ++i) {
      const double log_mh_ratio = proposal_likelihoods[i] - likelihoods[i];
      if (log_mh_ratio > 0.0 || distn->RandBernoulli(exp(log_mh_ratio))) {
        current_scores[i] = proposal_scores[i];
        likelihoods[i] = proposal_likelihoods[i];
        num_accept[i]++;
      }
    }
  }
  *final_scores = current_scores;
}

namespace poisson {
ScaledFeatureMhProposer::ScaledFeatureMhProposer(
    ArraySlice<pair<int, int>> level_posn_size,
    ArraySlice<double> aggregate_scaling,
    ArraySlice<double> prediction,
    ArraySlice<double> events,
    ArraySlice<double> proposal_sds,
    util::random::Distribution* distn)
    : level_posn_size_(level_posn_size),
      aggregate_scaling_(aggregate_scaling),
      prediction_(prediction),
      events_(events),
      proposal_sds_(proposal_sds),
      distn_(distn),
      current_level_index_(0) {
  CHECK_GT(level_posn_size_.size(), 0);
}

namespace {
  double functionWrapper(double x, void* params) {
    return static_cast<ScaledFeatureLoglik*>(params)
        ->EvaluateDx(x);
  }
}  // namespace

int ScaledFeatureRootSolver::Solve(double* root) {
  // returns -1 if Brent's method couldn't find a solution,
  // otherwise modifies the value at root and returns used iterations >= 0
  InitializeBracket(*root);

  double ub = ub_;
  double lb = lb_;
  if (ub == lb) {
    *root = ub;
    return 0;
  }

  // Brents method expects the root to be contained in an open interval
  ub += epsilon_;
  lb -= epsilon_;

  gsl_function function = {
    &functionWrapper,
    llik_
  };

  const gsl_root_fsolver_type* t;
  gsl_root_fsolver* s;
  t = gsl_root_fsolver_brent;
  s = gsl_root_fsolver_alloc(t);
  gsl_root_fsolver_set(s, &function, lb, ub);

  int status;
  int iter = 0;
  do {
    iter++;
    gsl_root_fsolver_iterate(s);
    lb = gsl_root_fsolver_x_lower(s);
    ub = gsl_root_fsolver_x_upper(s);
    // 0.005 is the absolute error,
    // this fixes a problem with high iterations for a root close to 0
    status = gsl_root_test_interval(lb, ub, 0.005, epsilon_);
  } while (status == GSL_CONTINUE && iter < max_brent_iter_);

  if (iter == max_brent_iter_ && status != GSL_SUCCESS) {
    LOG(INFO) << "Brent method did not find the root in the "
              << "interval [" << lb << ", " << ub << "] at iteration "
              << iter;
    return -1;
  }

  gsl_root_fsolver_free(s);
  *root = 0.5 * (ub + lb);
  return iter;
}

void ScaledFeatureRootSolver::InitializeBracket(double start) {
  // We know that the scaled features logLik is a concave function f(x),
  // hence for negative enough x < 0, the derivative is df/dx > 0. Also
  // for x > 0 large enough df/dx < 0. Bounds are needed to use Brent's
  // method for root finding.
  // TODO(kuehnelf): maybe find bounds analytically...
  auto sign = [](double val) -> int { return (val > 0) - (val < 0); };

  const int sign_llik_dx0 = sign(llik_->EvaluateDx(start));
  if (sign_llik_dx0 == 0) {
    ub_ = lb_ = start;
    return;
  }

  int sign_llik_dx;
  double delta;
  // TODO(kuehnelf): use grid search with analytical upper, lower bounds.
  delta = 0.05 * sign_llik_dx0;
  int i;
  for (i = 0; i < max_bound_iter_; ++i, delta *= 2.0) {
    sign_llik_dx = sign(llik_->EvaluateDx(start + delta));
    if (sign_llik_dx0 != sign_llik_dx) {
      break;
    }
  }

  // TODO(kuehnelf): this should not be an issue with analytical bounds
  if (i == max_bound_iter_ && sign_llik_dx0 == sign_llik_dx) {
      LOG(FATAL) << "Failed to find bounds for ML "
                 << "estimate after " << i << "iterations";
  }

  if (sign_llik_dx == 0) {
    lb_ = ub_ = start + delta;
  } else if (sign_llik_dx0 > 0) {
    ub_ = start + delta;
    lb_ = (i == 0) ? start : start + 0.5 * delta;
  } else if (sign_llik_dx0 < 0) {
    lb_ = start + delta;
    ub_ = (i == 0) ? start : start + 0.5 * delta;
  } else {
    LOG(FATAL) << "This case must never occur.";
  }
}

ScaledFeatureLoglik::ScaledFeatureLoglik(
    double inverse_variance,
    ArraySlice<pair<int, int>> level_posn_size,
    ArraySlice<double> aggregate_scaling,
    ArraySlice<double> prediction,
    ArraySlice<double> events)
    : inverse_variance_(inverse_variance),
      level_posn_size_(level_posn_size),
      aggregate_scaling_(aggregate_scaling),
      prediction_(prediction),
      events_(events),
      current_level_index_(-1) {
  this->IncrementLevelIndex();
}

void ScaledFeatureLoglik::IncrementLevelIndex() {
  current_level_index_++;
  if (current_level_index_ < level_posn_size_.size()) {
    const auto& position_size = level_posn_size_[current_level_index_];
    const int current_posn = position_size.first;
    const int current_size = position_size.second;
    current_prediction_ = ArraySlice<double>(prediction_,
                                             current_posn, current_size);
    current_events_ = ArraySlice<double>(events_,
                                         current_posn, current_size);
    current_scaling_ = ArraySlice<double>(aggregate_scaling_,
                                          current_posn, current_size);
  }
}

// Assumes that 'r' is non-NULL and has length at least that of 'x'
void ScaledFeatureLoglik::Evaluate(
    ArraySlice<double> x, MutableArraySlice<double> r) const {
  for (int i = 0; i < x.size(); ++i) {
    // The normal prior log likelihood
    double llik = -0.5 * inverse_variance_ * x[i] * x[i];

    // Pre-compute the exp() of the parameter which is re-used in the loop below
    const double x_cur = x[i];
    const int current_size = current_scaling_.size();
    for (int j = 0; j < current_size; ++j) {
      const double x_scale = x_cur * current_scaling_[j];
      const double p_events = current_prediction_[j];
      const double events = current_events_[j];
      llik += -p_events * exp(x_scale) + events * x_scale;
    }
    r[i] = llik;
  }
}

double ScaledFeatureLoglik::EvaluateDx(const double x) const {
  // From the normal prior log likelihood
  double llik_dx = -inverse_variance_ * x;

  const int current_size = current_scaling_.size();
  for (int j = 0; j < current_size; ++j) {
    const double scale = current_scaling_[j];
    const double scaled_p_events = current_prediction_[j] * exp(scale * x);
    llik_dx += (current_events_[j] - scaled_p_events) * scale;
  }
  return llik_dx;
}

}  // namespace poisson
}  // namespace emre
