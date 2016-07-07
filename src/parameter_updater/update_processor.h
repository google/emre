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

// The UpdateProcessor class computes statistics necessary to Gibbs sample
// or otherwise update the random effects, but it does not perform the sampling
// step itself.  This is actually delegated to the BlockRelaxation class which
// uses the statistics calculated by this class.
//
// Most of the updates make use of the current model prediction which is stored
// in a vector named 'p_events' below.  This prediction is based on _every_
// feature in the model.  The UpdateProcessor objects use these predictions
// as well as the input data (named 'obs' in the functions below) to get summary
// stats that determine posterior distributions are maximum likelihood updates.

#ifndef EMRE_PARAMETER_UPDATER_UPDATE_PROCESSOR_H_  // NOLINT
#define EMRE_PARAMETER_UPDATER_UPDATE_PROCESSOR_H_

#include "indexers/indexer.h"
#include "util/arrayslice.h"

namespace emre {

class UpdateProcessor {
 public:
  struct SupplementalStats {
    util::ArraySlice<double> aggregate_scaling;
    util::ArraySlice<pair<int, int>> level_posn_size;
  };

  UpdateProcessor() {}
  virtual ~UpdateProcessor() {}

  // The model fitting maintains a vector of 'predictions' (passed in through
  // 'p_events').  This function adds the features contribution to the final
  // prediction (through a product in a poisson model or by adding in a
  // gaussian model).
  virtual void AddToPrediction(IndexReader* index,
                               util::ArraySlice<double> coefficients,
                               util::MutableArraySlice<double> p_events) = 0;

  // Before updating model parameters through the fitting or sampling algorithm
  // we must compute the sufficient statistics as level_predicted_events and
  // level events (Poisson case). The immutable part of the sufficient stats,
  // level events is assumed to be already computed.
  //
  // It is assumed that level_predicted_events is properly sized,
  // already zero'ed out for the Poisson case, or properly initialized in the
  // Gaussian case!
  virtual void GetStatsForUpdate(
      IndexReader* index,
      util::ArraySlice<double> offsets,
      util::ArraySlice<double> coefficients,
      util::ArraySlice<double> p_events,
      util::MutableArraySlice<double> level_predicted_events) = 0;

  // Same as above, but level events is return in MutableArraySlice, and
  // previous level events cache is cleared if existed.
  //
  // level_events must either be properly sized or have size 0. When size is 0,
  // no output to level_events.

  // Most of time stats size equals num levels.
  virtual int GetStatsSize(int num_levels) {
    return num_levels;
  }

  // The model fitting maintains a vector of predictions (passed in through
  // 'p_events'). After a change in the parameters the predictions can be
  // updated through this function.
  virtual void UpdatePredictions(IndexReader* index,
                                 util::ArraySlice<double> coefficient_changes,
                                 util::MutableArraySlice<double> p_events) = 0;

  // Some prior updater require supplemental stats from the
  // update processor, i.e. scaled poisson features. Sub-classes should override
  // this method to return additional stats.
  virtual void PrepareUpdater(SupplementalStats* stats) {}
};

}  // namespace emre

#endif  // EMRE_PARAMETER_UPDATER_UPDATE_PROCESSOR_H_  // NOLINT
