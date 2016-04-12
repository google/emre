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

#include "scaled_poisson_block_relaxation.h"  // NOLINT

#include <math.h>

#include "contentads/analysis/caa/search_plus/regmh/emre/src/base/logging.h"
#include "contentads/analysis/caa/search_plus/regmh/emre/src/base/stringprintf.h"
#include "contentads/analysis/caa/search_plus/regmh/emre/src/parameter_updater/scaled_feature_util.h"

namespace emre {

using util::MutableArraySlice;

void ScaledPoissonOptimizer::UpdateRanefs(
    const FeatureFamilyPrior& prior,
    const UpdateParameters& update_parameters,
    MutableArraySlice<double> ranefs) {
  const int num_agg_levels = update_parameters.num_levels;
  const int num_levels = ranefs.size();
  CHECK_GE(num_agg_levels, num_levels);
  CHECK_GE(update_parameters.scores.size(), num_levels);
  auto events = update_parameters.auxiliary;
  auto p_events = update_parameters.predicted;
  CHECK_EQ(events.size(), p_events.size());
  // these parameters are only used for scaled poisson models
  const auto sstats = update_parameters.supplemental_stats;
  auto aggregate_scaling = sstats.aggregate_scaling;
  auto level_posn_size = sstats.level_posn_size;
  CHECK_GT(level_posn_size.size(), 0)
      << "we must have an aggregated scaling index";
  const double prior_inv_var = prior.inverse_variance();

  poisson::ScaledFeatureLoglik llik_cback(
      prior_inv_var, level_posn_size, aggregate_scaling, p_events, events);
  poisson::ScaledFeatureRootSolver solver(&llik_cback);

  double total_num_steps = 0.0;
  double* rptr = ranefs.data();
  for (int i = 0; i < num_levels; ++i, ++rptr) {
    llik_cback.IncrementLevelIndex();
    int curr_num_steps = solver.Solve(rptr);
    if (curr_num_steps == -1) {
      auto scores = ranefs.data();
      std::string dump_str = "";
      for (int j = 0; j < i; ++j) {
        StringAppendF(&dump_str, "%.0f, %.1f, %.3f, %.3f\n",
                      events[j], p_events[j], aggregate_scaling[j], scores[j]);
      }
      LOG(FATAL) << "Dumping scaled ranefs scores for debugging:\n"
                 << dump_str;
    }
    total_num_steps += curr_num_steps;
  }
  LOG(INFO) << "Average number of iteration steps for ML estimate: "
            << total_num_steps / num_levels;
}

ScaledPoissonGibbsSampler::ScaledPoissonGibbsSampler()
    : num_steps_per_iteration_(20) {
}

void ScaledPoissonGibbsSampler::UpdateRanefs(
    const FeatureFamilyPrior& prior,
    const UpdateParameters& update_parameters,
    MutableArraySlice<double> ranefs) {
  const int num_agg_levels = update_parameters.num_levels;
  const int num_levels = ranefs.size();
  CHECK_GE(num_agg_levels, num_levels);
  CHECK_GE(update_parameters.scores.size(), num_levels);
  auto events = update_parameters.auxiliary;
  auto p_events = update_parameters.predicted;
  CHECK_EQ(events.size(), p_events.size());
  // these parameters are only used for scaled poisson models
  const auto sstats = update_parameters.supplemental_stats;
  auto aggregate_scaling = sstats.aggregate_scaling;
  auto level_posn_size = sstats.level_posn_size;
  CHECK_GT(level_posn_size.size(), 0)
      << "we must have an aggregated scaling index";
  const double prior_inv_var = prior.inverse_variance();
  auto* rng = this->GetRng();

  // We have no closed from solution for the sampling from the ranef posterior.
  // Instead we resort to use MonteCarlo sampling, which currently is
  // MetropolisHastings (MH) with an initial proposal normal distribution:
  if (proposal_sds_.size() != num_levels) {
    proposal_sds_.assign(num_levels, 5.0);
    acceptance_counts_.assign(num_levels, {0, 0});
  }

  // burnin stats for logging
  int count_increase_sds = 0;
  int count_decrease_sds = 0;

  poisson::ScaledFeatureMhProposer proposer(
      level_posn_size, aggregate_scaling, p_events, events,
      proposal_sds_, rng);

  poisson::ScaledFeatureLoglik llik_cback(
      prior_inv_var, level_posn_size, aggregate_scaling, p_events, events);

  for (int i = 0; i < num_levels; ++i) {
    int num_accepted = 0;
    ranefs[i] = metropolis_hastings::RunMetropolisHastings(
        llik_cback, update_parameters.scores[i], num_steps_per_iteration_,
        &proposer, rng, &num_accepted);

    proposer.IncrementLevelIndex();
    llik_cback.IncrementLevelIndex();

    auto* ct = &(acceptance_counts_[i]);
    ct->first += num_steps_per_iteration_;
    ct->second += num_accepted;

    if (ct->second <= 0.2 * ct->first) {
      proposal_sds_[i] *= 0.5;
      ++count_decrease_sds;
      ct->first = 0;
      ct->second = 0;
    } else if (ct->second >= 0.8 * ct->first) {
      proposal_sds_[i] *= 2.0;
      ++count_increase_sds;
      ct->first = 0;
      ct->second = 0;
    }
  }

  LOG(INFO) << "Proposal sd adjustments for " << ranefs.size()
      << " features, " << count_increase_sds
      << " times increased sds and " << count_decrease_sds
      << " decreased sds.";
}

}  // namespace emre
