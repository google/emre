.IsRanefUpdater <- function(object) {
  # A helper function
  return(inherits(object, c("Rcpp_RanefUpdater",
                            "Rcpp_GaussianRanefUpdater",
                            "Rcpp_ScaledPoissonRanefUpdater")))
}

.CreateRanefUpdater <- function(feature.family.pb, index.reader) {
  # Creates a new random effects updater object.
  #
  # Args:
  #   feature.family.pb: A FeatureFamilyPrior protobuffer
  #   indexer.reader: An S4 object with the index reader
  #
  # Returns:
  #   A ranef updater handle with the RanefUpdater object
  stopifnot(typeof(index.reader) == "externalptr",
            inherits(index.reader, "IndexReader"),
            inherits(feature.family.pb, c("Message", "RProtoBuf")))

  # create a new ranef updater object
  ff.pb <- paste0(feature.family.pb)
  updater <- ranef.updater.module$RanefUpdater
  updater.handle <- new(updater, ff.pb, index.reader)

  return(updater.handle)
}

.CreateGaussianRanefUpdater <- function(feature.family.pb, index.reader) {
  # Creates a new gaussian random effects updater object.
  #
  # Args:
  #   feature.family.pb: A FeatureFamilyPrior protobuffer
  #   indexer.reader: An S4 object with the index reader
  #
  # Returns:
  #   A ranef updater handle with the GaussianRanefUpdater object
  stopifnot(typeof(index.reader) == "externalptr",
            inherits(index.reader, "IndexReader"),
            inherits(feature.family.pb, c("Message", "RProtoBuf")))

  # create a new ranef updater object
  ff.pb <- paste0(feature.family.pb)
  updater <- ranef.updater.module$GaussianRanefUpdater
  updater.handle <- new(updater, ff.pb, index.reader)
  return(updater.handle)
}

.CreateScaledPoissonRanefUpdater <- function(feature.family.pb, index.reader) {
  # Creates a new scaled poisson random effects updater object.
  #
  # Args:
  #   feature.family.pb: A FeatureFamilyPrior protobuffer
  #   indexer.reader: An S4 object with the index reader
  #
  # Returns:
  #   A ranef updater handle with the ScaledPoissonRanefUpdater object
  stopifnot(typeof(index.reader) == "externalptr",
            inherits(index.reader, "IndexReader"),
            inherits(feature.family.pb, c("Message", "RProtoBuf")))

  # create a new ranef updater object
  ff.pb <- paste0(feature.family.pb)
  updater <- ranef.updater.module$ScaledPoissonRanefUpdater
  updater.handle <- new(updater, ff.pb, index.reader)
  return(updater.handle)
}

.GetIndexReader <- function(updater.handle) {
  # Returns an S4 object with the index reader
  #
  # Args:
  #   updater.handle: The RanefUpdater object
  # Returns:
  #    An S4 object with the index reader
  if (!.IsRanefUpdater(updater.handle)) {
    stop("The updater handle must inherit Rcpp_RanefUpdater. Received ",
         "class(updater.handle) = ", class(updater.handle))
  }

  return(updater.handle$get.index.reader())
}

.AddToPrediction <- function(updater.handle, coefficients, p.events) {
  # Adds the scores for this feature to the predicted events for each
  # observation.  This function modifies p.events in place and for
  # convenience also returns it.
  #
  # Args:
  #   updater.handle: The RanefUpdater object
  #   coefficients: A numeric vector with the scores for this feature family
  #   p.events: A numeric vector with the predicted events for each observation
  #
  # Returns:
  #   Modifies in-place the predicted events per observations and returns it.
  stopifnot(.IsRanefUpdater(updater.handle),
            is.numeric(coefficients),
            is.numeric(p.events))

  updater.handle$inp.addto.prediction(coefficients, p.events)
}

.CollectStats <- function(updater.handle, offset,
                          coefficients, p.events,
                          prediction = NULL) {
  # Aggregates predicted events p.events to the family feature level.
  #
  # Args:
  #   updater.handle: The RanefUpdater object
  #   offset: A numeric vector with the offset
  #   coefficients: A numeric vector with the feature family coefficients
  #   p.events: A numeric vector with the predicted events for each
  #     observation
  #   prediction: A numeric vector in which to store the predicted events
  #     per feature level. If this is null a new numeric vector for the
  #     prediction will be created and returned. Otherwise, this vector
  #     will be modified in place.
  # Returns:
  #   A numeric vectors, with the predicted events per level.
  stopifnot(.IsRanefUpdater(updater.handle),
            is.numeric(offset),
            is.numeric(coefficients),
            is.numeric(p.events),
            is.numeric(prediction) || is.null(prediction))

  if (is.null(prediction) || length(prediction) < length(coefficients)) {
    prediction <- rep(0.0, length(coefficients))
  }

  updater.handle$inp.collect.stats(offset, coefficients, p.events, prediction)
}

.UpdatePrediction <- function(updater.handle, p.events, coefficient.ratios) {
  # Updates the predicted events for each observation with the the
  # coefficient ratios (new coefficients / old coefficients)
  #
  # Args:
  #   p.events: A numeric vector with the predicted events for each
  #     observation
  #   updater.handle: The RanefUpdater object
  #   coefficient.ratios: A numeric vector with new / old
  #     coefficients for this feature family
  # Returns:
  #   Modifies in-place the predicted events and returns it.
  stopifnot(.IsRanefUpdater(updater.handle),
            is.numeric(coefficient.ratios),
            is.numeric(p.events))

  updater.handle$inp.update.prediction(p.events, coefficient.ratios)
}

.UpdateCoefficients <- function(updater.handle, coefficients,
                                prediction, events) {
  # Updates coefficients for each feature level by drawing from a conditional
  # distribution or returning the mode of this distribution. Uses the sufficient
  # prediction and events per level statistics.
  #
  # Args:
  #   updater.handle: The RanefUpdater object
  #   coefficients: A numeric vector with the feature family coefficients
  #   prediction: A numeric vector with the predictions for each level
  #   events: A numeric vector with the events for each level
  # Returns:
  #   Modifies in-place the coefficient vector and returns it
  stopifnot(.IsRanefUpdater(updater.handle),
            is.numeric(coefficients),
            is.numeric(prediction),
            is.numeric(events))

  updater.handle$inp.update.ranef.coefficients(coefficients, prediction, events)
}

.UpdateRanefPrior <- function(updater.handle, coefficients,
                              prediction, events) {
  # Based on current predicted and actual events per feature level and provided
  # scores, updates the random effects prior via an EM step.
  #
  # Args:
  #   updater.handle: The RanefUpdater object
  #   coefficients: A numeric vector with the scores for this family
  #   prediction: A numeric vector with the predictions for each level
  #   events: A numeric vector with the events for each level
  # Returns:
  #   NULL
  stopifnot(.IsRanefUpdater(updater.handle),
            is.numeric(coefficients),
            is.numeric(prediction),
            is.numeric(events))

  updater.handle$update.ranef.prior(coefficients, prediction, events)
}

.GetRanefPrior <- function(updater.handle) {
  # Returns the current prior protobuffer
  #
  # Args:
  #   updater.handle: The RanefUpdater object
  # Returns:
  #   The current prior protobuffer
  stopifnot(.IsRanefUpdater(updater.handle))

  prior.pb <- updater.handle$get.ranef.prior()
  return(readASCII(emre.FeatureFamilyPrior, prior.pb))
}
