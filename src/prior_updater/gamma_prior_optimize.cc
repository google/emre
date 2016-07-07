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

#include "gamma_prior_optimize.h"  // NOLINT

#include <algorithm>

#include "base/logging.h"
#include "util/emre_util.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_sf_psi.h"
#include "likelihoods.h"  // NOLINT
#include "metropolis_hastings.h"  // NOLINT

namespace emre {

using util::ArraySlice;
using util::EmreUtil;
using util::MutableArraySlice;

RealFunctionInLogSpace::RealFunctionInLogSpace(
    RealFunction* cback, bool should_own)
    : cback_(cback), should_own_(should_own) {
  CHECK_NOTNULL(cback_);
}

RealFunctionInLogSpace::~RealFunctionInLogSpace() {
  if (should_own_) {
    delete cback_;
  }
  cback_ = nullptr;
}

// Assumes that 'r' is non-NULL and has length at least that of 'x'
void RealFunctionInLogSpace::Evaluate(ArraySlice<double> x,
                                      MutableArraySlice<double> r) const {
  vector<double> exp_x_buffer(x.size(), 0.0);
  std::transform(x.begin(), x.end(), exp_x_buffer.begin(), exp);
  cback_->Evaluate(exp_x_buffer, r);
}


class EmreParamLikelihoodRB : public RealFunction {
 public:
  EmreParamLikelihoodRB(
      const pair<double, double>& prior,
      ArraySlice<double> p_events,
      ArraySlice<double> events);

  virtual ~EmreParamLikelihoodRB() {}

  void Evaluate(ArraySlice<double> x,
                MutableArraySlice<double> r) const override;

 private:
  pair<double, double> model_prior_;
  vector<pair<double, double>> posterior_mean_;
};

// This struct is used by NonNegativeOptimize to hold parameter values and
// the associated likelihoods.
struct NonNegativeOptimizePair {
  NonNegativeOptimizePair(double x_input, double value_input)
      : x(x_input), value(value_input) {}

  bool operator<(const NonNegativeOptimizePair& v) const {
    return x < v.x;
  }

  double x;
  double value;
};

void NonNegativeOptimize(const RealFunction& cback,
                         double initial_value,
                         int grid_size, int num_zoom,
                         vector<double>* evaluation_points,
                         vector<double>* values,
                         int* max_index) {
  static const double kStepSize = 0.7;

  vector<NonNegativeOptimizePair> evaluations;

  for (int zoom_idx = 0; zoom_idx < num_zoom; ++zoom_idx) {
    vector<double> new_evaluation_points;
    if (zoom_idx == 0) {
      const double scale = grid_size + 1 + (grid_size % 2);

      for (int i = 0; i < grid_size; ++i) {
        const double mlt = static_cast<double>(2 * i + 1) / scale;
        new_evaluation_points.push_back(initial_value * mlt);
      }
    } else {
      if (*max_index == 0) {
        // Note that although this points are added in descending order, they
        // are sorted in the block below.
        double push_value = evaluations[0].x * kStepSize;
        for (int i = 0; i < grid_size; ++i) {
          new_evaluation_points.push_back(push_value);
          push_value *= 0.7;
        }
      } else if (*max_index == evaluations.size() - 1) {
        double push_value = evaluations[evaluations.size() - 1].x / kStepSize;
        for (int i = 0; i < grid_size; ++i) {
          new_evaluation_points.push_back(push_value);
          push_value /= kStepSize;
        }

      } else {
        const double min_value = evaluations[*max_index - 1].x;
        const double max_value = evaluations[*max_index + 1].x;
        for (int i = 0; i < grid_size; ++i) {
          const double r = static_cast<double>(i + 1) / (grid_size + 2);
          new_evaluation_points.push_back(min_value * r
                                          + max_value * (1.0 - r));
        }
      }
    }

    CHECK_LT(0, new_evaluation_points.size());

    vector<double> new_values(new_evaluation_points.size());
    cback.Evaluate(new_evaluation_points, &new_values);

    for (int i = 0; i < new_evaluation_points.size(); ++i) {
      evaluations.push_back(NonNegativeOptimizePair(new_evaluation_points[i],
                                                    new_values[i]));
    }
    std::sort(evaluations.begin(), evaluations.end());

    *max_index = 0;
    double max_value = evaluations[0].value;
    for (int i = 0; i < evaluations.size(); ++i) {
      if (evaluations[i].value > max_value) {
        max_value = evaluations[i].value;
        *max_index = i;
      }
    }
  }

  evaluation_points->clear();
  values->clear();
  for (int i = 0; i < evaluations.size(); ++i) {
    evaluation_points->push_back(evaluations[i].x);
    values->push_back(evaluations[i].value);
  }
}

namespace {

double UpdateGammaPoissonVarianceEstimatesSample(
    const pair<double, double>& prior, ArraySlice<double> samples,
    int grid_size, int num_zoom) {
  vector<double> gamma_params;
  vector<double> likelihoods;
  int max_index = 0;

  EmreSampleParamLikelihood cback(samples);
  NonNegativeOptimize(cback, prior.second, grid_size, num_zoom,
                      &gamma_params, &likelihoods, &max_index);
  return gamma_params[max_index];
}

double SampleGammaPoissonVarianceEstimatesIntegrated(
    const pair<double, double>& prior,
    ArraySlice<double> p_events,
    ArraySlice<double> events,
    int grid_size, int num_steps, bool die_on_failure,
    util::random::Distribution* distn) {
  RealFunctionInLogSpace logspace_llik(
      new EmreParamIntegratedLikelihood(p_events, events),
      true /* should_own */);

  // First we are going to look at the likelihood on a grid to find a window
  // around the current point (prior.second) such that the likelihood does not
  // very too much.  This is not foolproof; it only looks at the values at
  // some points on a grid.
  double max_log_scaling = log(2.0);
  double proposal_sd = 0.0;
  const double log_current_prior = log(prior.second);
  for (int i = 0; i < 10; ++i) {
    pair<double, double> search_range(log_current_prior - max_log_scaling,
                                      log_current_prior + max_log_scaling);
    pair<double, double> range = FindLikelihoodRangeForMCMC(
        logspace_llik, log_current_prior /* log_initial_value */,
        grid_size, 2.0 /* max_llik_delta */, search_range);
    proposal_sd = 0.5 * std::max(range.second - log_current_prior,
                                 log_current_prior - range.first);
    CHECK_GE(proposal_sd, 0.0);
    proposal_sd = std::max(proposal_sd, 0.0);
    if (proposal_sd <= 0.0) {
      LOG(INFO) << "Failed to find proposal Std Dev in range "
                << exp(max_log_scaling);
    } else {
      break;
    }
    max_log_scaling *= 0.5;
  }
  CHECK(proposal_sd > 0.0 || !die_on_failure);
  if (proposal_sd <= 0.0) {
    return prior.second;
  }
  LOG(INFO) << "proposal_sd = " << proposal_sd;

  metropolis_hastings::SymmetricGaussianMhProposer
      move_generator(proposal_sd, distn);

  int num_accepted = -1;
  const double log_gamma_param =
      metropolis_hastings::RunMetropolisHastings<double>(
          logspace_llik, log_current_prior /* initial_value */, num_steps,
          &move_generator, distn, &num_accepted);

  const double acceptance_rate =
      static_cast<double>(num_accepted) / static_cast<double>(num_steps);
  LOG(INFO) << "accepted " << num_accepted << " of "
            << num_steps << " steps " << "(" << 100.0 * acceptance_rate << "%)";
  return exp(log_gamma_param);
}

double UpdateGammaPoissonVarianceEstimatesIntegrated(
    const pair<double, double>& prior,
    ArraySlice<double> p_events,
    ArraySlice<double> events,
    int grid_size, int num_zoom) {
  vector<double> gamma_params;
  vector<double> likelihoods;
  int max_index = 0;

  EmreParamIntegratedLikelihood cback(p_events, events);
  NonNegativeOptimize(cback, prior.second, grid_size, num_zoom,
                      &gamma_params, &likelihoods, &max_index);
  return gamma_params[max_index];
}

double UpdateGammaPoissonVarianceEstimatesRB(
    const pair<double, double>& prior,
    ArraySlice<double> p_events,
    ArraySlice<double> events,
    int grid_size, int num_zoom) {
  vector<double> gamma_params;
  vector<double> likelihoods;
  int max_index = 0;

  EmreParamLikelihoodRB cback(prior, p_events, events);
  NonNegativeOptimize(cback, prior.second, grid_size, num_zoom,
                      &gamma_params, &likelihoods, &max_index);
  return gamma_params[max_index];
}

}  // namespace


// This function is used in MCMC sampling of prior parameters.  It searches
// in the range
//    [x0 - max_log_scaling, x0 + max_log_scaling]
// (where x0 = log_initial_value) for an interval such that both end-points
// are no more than 'max_llik_delta' lower than cback(exp(initial_value)).
// in likelihood.
//
// Returns the width 'R', so if max_llik_delta = 2 then R * 0.5 would be a
// reasonable choice of proposal standard deviation in Metropolis Hastings.
//
// max_llik_delta should be non-negative and the function will check for this.

pair<double, double> FindLikelihoodRangeForMCMC(
    const RealFunction& cback, double initial_value, int grid_size,
    double max_llik_delta, const pair<double, double>& range) {
  CHECK_GT(grid_size, 1);
  CHECK_GE(max_llik_delta, 0.0);
  CHECK_GE(range.second, range.first);

  vector<double> new_evaluation_points = {initial_value};
  vector<double> scalings = {0.0};

  const double step_size = ((range.second - range.first)
                            / static_cast<double>(grid_size - 1));
  double current_point = range.first;
  for (int i = 0; i < grid_size; ++i) {
    new_evaluation_points.push_back(current_point);
    current_point += step_size;
  }

  vector<double> new_values(new_evaluation_points.size());
  cback.Evaluate(new_evaluation_points, &new_values);

  const double initial_llik = new_values[0];
  const double lb = initial_llik - max_llik_delta;

  pair<double, double> final_range(initial_value, initial_value);

  // We skip i = 0 because this index held 'initial_value'.
  for (int i = 1; i < new_evaluation_points.size(); ++i) {
    if (new_values[i] >= lb) {
      if (new_evaluation_points[i] < initial_value) {
        final_range.first = std::min(final_range.first,
                                     new_evaluation_points[i]);
      } else {
        final_range.second = std::max(final_range.second,
                                      new_evaluation_points[i]);
      }
    }
  }
  return final_range;
}

void GammaPoissonPriorIntegratedOptimize::UpdateVariance(
    ArraySlice<double> stats_events,
    ArraySlice<double> stats_p_events,
    ArraySlice<double> coefficients,
    util::random::Distribution* rng) {
  int num_levels = coefficients.size();
  CHECK_GE(stats_events.size(), num_levels);
  CHECK_GE(stats_p_events.size(), num_levels);
  ArraySlice<double> p_events(stats_p_events, 0, num_levels);
  ArraySlice<double> events(stats_events, 0, num_levels);
  vector<double> p_events_subset;
  vector<double> events_subset;

  const int max_levels = this->GetMaxLevels();
  if (max_levels > 0 && max_levels < num_levels) {
    p_events_subset.resize(max_levels, 0.0);
    events_subset.resize(max_levels, 0.0);
    EmreUtil::GetVectorSubset(p_events, &p_events_subset);
    EmreUtil::GetVectorSubset(events, &events_subset);

    p_events = p_events_subset;
    events = events_subset;
  }

  const double gamma_param = UpdateGammaPoissonVarianceEstimatesIntegrated(
      this->GetGammaPrior(),
      p_events, events,
      this->GetPriorOptimConfig().grid_size,
      this->GetPriorOptimConfig().num_iterations);
  this->SetGammaPrior(gamma_param, gamma_param);
}

void GammaPoissonPriorSampleOptimize::UpdateVariance(
    ArraySlice<double> stats_events,
    ArraySlice<double> stats_p_events,
    ArraySlice<double> coefficients,
    util::random::Distribution* rng) {
  const double fitted_gamma_param = UpdateGammaPoissonVarianceEstimatesSample(
      this->GetGammaPrior(), coefficients,
      this->GetPriorOptimConfig().grid_size,
      this->GetPriorOptimConfig().num_iterations);
  this->SetGammaPrior(fitted_gamma_param, fitted_gamma_param);
}

void GammaPoissonPriorRaoBlackwellizedOptimize::UpdateVariance(
    ArraySlice<double> stats_events,
    ArraySlice<double> stats_p_events,
    ArraySlice<double> coefficients,
    util::random::Distribution* rng) {
  int num_levels = coefficients.size();
  CHECK_GE(stats_events.size(), num_levels);
  CHECK_GE(stats_p_events.size(), num_levels);
  ArraySlice<double> p_events(stats_p_events, 0, num_levels);
  ArraySlice<double> events(stats_events, 0, num_levels);

  const double gamma_param = UpdateGammaPoissonVarianceEstimatesRB(
      this->GetGammaPrior(),
      p_events, events,
      this->GetPriorOptimConfig().grid_size,
      this->GetPriorOptimConfig().num_iterations);
  this->SetGammaPrior(gamma_param, gamma_param);
}

void GammaPoissonPriorIntegratedGibbsSampler::UpdateVariance(
    ArraySlice<double> stats_events,
    ArraySlice<double> stats_p_events,
    ArraySlice<double> coefficients,
    util::random::Distribution* rng) {
  int num_levels = coefficients.size();
  CHECK_GE(stats_events.size(), num_levels);
  CHECK_GE(stats_p_events.size(), num_levels);
  ArraySlice<double> events(stats_events, 0, num_levels);
  ArraySlice<double> p_events(stats_p_events, 0, num_levels);
  const double fitted_gamma_param =
      SampleGammaPoissonVarianceEstimatesIntegrated(
          this->GetGammaPrior(),
          p_events, events,
          this->GetPriorOptimConfig().grid_size,
          this->GetPriorOptimConfig().num_iterations /* num_steps */,
          false /* die_on_failure */, rng);
  this->SetGammaPrior(fitted_gamma_param, fitted_gamma_param);
}

void GammaPoissonPriorUpdater::SetProtoFromPrior(FeatureFamilyPrior* pb) const {
  CHECK_NOTNULL(pb);
  CHECK_GT(alpha_, 0.0);
  CHECK_GT(beta_, 0.0);
  pb->set_inverse_variance(beta_ * beta_ / alpha_);
  pb->set_mean(alpha_ / beta_);
  pb->set_max_levels_for_update(max_levels_for_update_);
}

void GammaPoissonPriorUpdater::SetPriorFromProto(const FeatureFamilyPrior& pb) {
  CHECK_GT(pb.mean(), 0.0);
  CHECK_GT(pb.inverse_variance(), 0.0);
  beta_ = pb.inverse_variance() * pb.mean();
  alpha_ = pb.mean() * beta_;
  max_levels_for_update_ = pb.max_levels_for_update();
}

EmreParamLikelihoodRB::EmreParamLikelihoodRB(
    const pair<double, double>& prior,
    ArraySlice<double> p_events,
    ArraySlice<double> events) {
  const int n_groups = events.size();
  model_prior_ = prior;
  posterior_mean_.resize(n_groups, pair<double, double>(0.0, 0.0));

  for (int i = 0; i < n_groups; ++i) {
    const double alpha_posterior = model_prior_.first + events[i];
    const double beta_posterior = model_prior_.second + p_events[i];
    const double digamma_alpha = gsl_sf_psi(alpha_posterior);
    const double log_beta = log(beta_posterior);

    posterior_mean_[i].first = alpha_posterior / beta_posterior;
    posterior_mean_[i].second = digamma_alpha - log_beta;
  }
}

void EmreParamLikelihoodRB::Evaluate(
    ArraySlice<double> x, MutableArraySlice<double> r) const {
  CHECK_GE(r.size(), x.size());
  const int n_groups = posterior_mean_.size();

  for (int i = 0; i < x.size(); ++i) {
    const double alphabeta = x[i];
    r[i] = n_groups * (alphabeta * log(alphabeta) - gsl_sf_lngamma(alphabeta));
  }

  // posterior_mean_.first = E[X|...]
  // posterior_mean_.second = E[log(X)|...]
  for (auto& mn : posterior_mean_) {
    for (int i = 0; i < x.size(); ++i) {
      r[i] += (x[i] - 1.0) * mn.second - x[i] * mn.first;
    }
  }
}

void EmreSampleParamLikelihood::Evaluate(
    ArraySlice<double> x, MutableArraySlice<double> r) const {
  CHECK_GE(r.size(), x.size());
  const double n_groups = this->GetSampleParams().size();
  for (int i = 0; i < x.size(); ++i) {
    r[i] = n_groups * (x[i] * log(x[i]) - gsl_sf_lngamma(x[i]));
  }

  for (auto sample_param : this->GetSampleParams()) {
    const double log_sample_param = log(sample_param);
    for (int i = 0; i < x.size(); ++i) {
      r[i] += (x[i] - 1.0) * log_sample_param - x[i] * sample_param;
    }
  }
}

// Static
void EmreParamIntegratedLikelihood::FillLookupTable(
    ArraySlice<double> x, MutableArraySlice<double> r, int num_elts) {
  const int x_size = x.size();
  for (int k = 0; k < num_elts; ++k) {
    for (int i = 0; i < x_size; ++i) {
      r[k * x_size + i] = gsl_sf_lngamma(x[i] + k);
    }
  }
}

void EmreParamIntegratedLikelihood::Evaluate(
    ArraySlice<double> x, MutableArraySlice<double> r) const {
  // The function to be implemented here is:
  // \sum_{j=1}^{num_levels} \left(
  //    x_i \log{x_i} - \lngamma{x_i}
  //  -(y_j + x_i)\log{p_j + x_i} + \lngamma{y_j + x_i}
  // where y_j is the number of events (conversions) for level j,
  // p_j = number of predicted events for level j and
  // x_i the variance from the distribution Gamma(x_i, x_i).
  // The function above represent the x_i dependent terms of the
  // data log likelihood log(p(y_j | p_j, x_i))
  const int num_elts = this -> NumElts();
  const int x_size = x.size();
  for (int i = 0; i < x_size; ++i) {
    r[i] = likelihoods::GammaNormLogConst(x[i], x[i]) * num_elts;
  }

  // The values in the lookup table depend on the input 'x', so this cannot
  // be moved to the constructor
  static const int kNumElements = 10;
  vector<double> lgamma_lookup_table(kNumElements * x.size(), 0.0);
  EmreParamIntegratedLikelihood::FillLookupTable(
      x, &lgamma_lookup_table, kNumElements);

  for (int j = 0; j < num_elts; ++j) {
    const double p_events = predicted_[j];
    const double events = ancillary_[j];
    if (events < (kNumElements - 1) && fabs(round(events) - events) < 1e-4) {
      const int ct = round(events);
      double* tbl = &(lgamma_lookup_table[ct * x_size]);
      for (int i = 0; i < x_size; ++i, ++tbl) {
        r[i] -= (x[i] + events) * log(x[i] + p_events) - *tbl;
      }
    } else {
      for (int i = 0; i < x_size; ++i) {
        r[i] -= likelihoods::GammaNormLogConst(x[i] + events, x[i] + p_events);
      }
    }
  }
}

}  // namespace emre
