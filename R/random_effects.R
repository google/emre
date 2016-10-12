RandomEffect <- R6Class("RandomEffect",
  cloneable = FALSE,
  public = list(
    coefficients = c(),
    events.per.level = c(),

    # constructor
    initialize = function(prior, index.reader) {
      stopifnot(!is.null(prior),
                !is.null(index.reader))
      if (prior$prior_update_type == 999) {
        private$is.fixed.effect <- TRUE
      }
      num.levels <- .IndexerNumLevels(index.reader)
      ff.name <- prior$feature_family
      EmreDebugPrint(
        sprintf("initialize poisson %s with %d levels", ff.name, num.levels))
      private$updater <- .CreateRanefUpdater(prior, index.reader)
      EmreDebugPrint("... intialized")
      self$coefficients <- rep(1.0, num.levels)
    },

    # public methods
    get.family.name = function() self$get.prior()$feature_family,
    get.feature.levels = function() {
      .IndexerStringLevels(.GetIndexReader(private$updater))
    },
    get.num.levels = function() { length(self$coefficients) },
    get.prior = function() { .GetRanefPrior(private$updater) },
    does.update.prior = function() { !private$is.fixed.effect },

    calc.immutable.stats = function(response, offset) {
      # Before updating model parameters through the fitting or sampling
      # algorithm, we must compute the sufficient statistics, here
      # aggregate (sum) events per level (the immutable part) and
      # the predicted events per level (mutable part).
      # This function computes only the immutable part and caches it in
      # self$events.per.level
      row2level.map <- .IndexerRowToLevelMap(.GetIndexReader(private$updater))
      # a negative index indicates that there a NA features
      idx <- (row2level.map < 0)
      row2level.map[idx] <- NA
      # this aggregates the response variable to the level id, leaving out NAs
      self$events.per.level <- tapply(response, row2level.map, FUN = sum)
      return(self$events.per.level)
    },

    add.to.prediction = function(p.events) {
      # Compute prediction with the factors of this family and modifies
      # p.events in place without copy.
      #
      # Args:
      #   p.events: R vector for predicted events (Poisson model) with
      #     length as the number of observations.
      #
      # Returns:
      #   p.events: R vector with adjusted predictions.
      .AddToPrediction(private$updater, self$coefficients, p.events)
    },

    collect.stats = function(p.events, offset) {
      # TODO(kuehnelf): last argument should be named parameter
      private$prediction.per.level <- .CollectStats(private$updater, offset,
                                                    self$coefficients, p.events,
                                                    NULL)
    },

    update.coefficients = function(p.events) {
      # Updates and returns p.events in place without a copy. Also updates
      # the member self$coefficients in place.
      #
      # Args:
      #   p.events: R vector for predicted events (Poisson model) with
      #     length as the number of observations.
      #
      # Returns:
      #   p.events: The same R vector (not a copy) with adjusted predictions.
      if (is.na(private$prediction.per.level) ||
          length(self$events.per.level) < length(self$coefficients)) return
      old.coefficients <- c(self$coefficients)  # make a copy
      .UpdateCoefficients(private$updater,
                          self$coefficients,
                          private$prediction.per.level,
                          self$events.per.level)

      coefficient.changes <- self$coefficients / old.coefficients
      .UpdatePrediction(private$updater, p.events, coefficient.changes)
    },

    update.prior = function() {
      # Returns: NULL
      if (is.na(private$prediction.per.level) ||
          length(self$events.per.level) < length(self$coefficients)) return
      # TODO(kuehnelf): improve API for MH sampling prior in R code
      .UpdateRanefPrior(private$updater,
                        self$coefficients,
                        private$prediction.per.level,
                        self$events.per.level)
    }),

    private = list(
      updater = NA,
      prediction.per.level = NA,
      # TODO(kuehnelf): we should push this into FeatureFamilyPrior
      is.fixed.effect = FALSE
    )
)

ScaledRandomEffect <- R6Class("ScaledRandomEffect",
  inherit = RandomEffect,
  cloneable = FALSE,
  public = list(
    initialize = function(prior, index.reader) {
      stopifnot(!is.null(prior),
                !is.null(index.reader))
      # TODO(kuehnelf): eliminate this section,
      # 999 is the don't update prior in the FeatureFamilyPrior proto
      if (prior$prior_update_type == 999) {
        private$is.fixed.effect <- TRUE
      }
      num.levels <- .IndexerNumLevels(index.reader)
      ff.name <- prior$feature_family
      EmreDebugPrint(
        sprintf("initialize scaled Poisson %s with %d levels",
                ff.name, num.levels))
      private$updater <- .CreateScaledPoissonRanefUpdater(prior, index.reader)
      EmreDebugPrint("... intialized")
      self$coefficients <- rep(0.0, num.levels)
    },

    calc.immutable.stats = function(response = response,
                                    offset = offset) {
      # TODO(kuehnelf):
      # we must take into account the unique level scaling map!
      ff.name <- self$get.family.name()
      stop(paste("Feature '", ff.name, "' is considered as a numeric one. ",
                 "Currently, Poisson models don't support numeric features. ",
                 "Formulas of the form '+ ", ff.name, " + ...' and ",
                 "'(0+'", ff.name, "' |re) ' will be supported ",
                 "in future versions.", sep = ""))
    },

    update.coefficients = function(p.events) {
      # Updates and returns p.events in place without copy. Also updates
      # the member self$coefficients in place.
      #
      # Args:
      #   p.events: R vector for predicted events (Poisson model) with
      #     length as the number of observations.
      #
      # Returns:
      #   p.events: The same R vector (not a copy) with adjusted predictions.
      if (is.na(private$prediction.per.level) ||
          length(self$events.per.level) < length(self$coefficients)) return
      old.coefficients <- c(self$coefficients)  # make a copy
      .UpdateCoefficients(private$updater, self$coefficients,
                          private$prediction.per.level,
                          self$events.per.level)

      coefficient.changes <- self$coefficients - old.coefficients
      return(.UpdatePrediction(private$updater, p.events, coefficient.changes))
    }
  )
)

GaussianRandomEffect <- R6Class("GaussianRandomEffect",
  inherit = RandomEffect,
  cloneable = FALSE,
  public = list(
    cached.immutable.stats = c(),
    invvar.per.level = c(),
    # constructor
    initialize = function(prior, index.reader) {
      stopifnot(!is.null(prior),
                !is.null(index.reader))
      if (prior$prior_update_type == 999) {
        private$is.fixed.effect <- TRUE
      }
      num.levels <- .IndexerNumLevels(index.reader)
      ff.name <- prior$feature_family
      EmreDebugPrint(
        sprintf("initialize gaussian %s with %d levels",
                ff.name, num.levels))
      private$updater <- .CreateGaussianRanefUpdater(prior, index.reader)
      private$residual.inv.var.callback <- function() { return(1.0) }
      EmreDebugPrint("... intialized")
      self$coefficients <- rep(0.0, num.levels)
    },

    calc.immutable.stats = function(response, inverse.variance) {
      # For gaussian model, part of the immutable stats is the product
      # of response, offsets and scaling aggregated to the feature level.
      # This quantity is used in the collect.stats function.
      # Another quantity is the product of inverse variance with the square
      # or the scaling. This quantity is used in the both the update.prior
      # and update coefficients function.
      row2level.map <- .IndexerRowToLevelMap(.GetIndexReader(private$updater))
      # a negative index indicates that there a NA features
      idx <- (row2level.map < 0)
      row2level.map[idx] <- NA
      # this aggregates the product of response, inverse.variance and
      # assuming scaling = 1.0 to the level id while leaving out NAs
      riv <- response * inverse.variance
      self$cached.immutable.stats <- tapply(riv, row2level.map, FUN = sum)
      # this is what we call invvar in the paper
      self$invvar.per.level <- tapply(inverse.variance, row2level.map,
                                      FUN = sum)
      return(self$cached.immutable.stats)
    },

    collect.stats = function(p.values, inverse.variance) {
      private$prediction.per.level <- c(self$cached.immutable.stats)  # copy
      # TODO(kuehnelf): rename prediction.per.level to error.per.level
      private$prediction.per.level <- .CollectStats(
          private$updater, inverse.variance,
          self$coefficients, p.values,
          private$prediction.per.level)
      residual.inv.variance <- private$residual.inv.var.callback()
      private$prediction.per.level <- private$prediction.per.level *
                                      residual.inv.variance
    },

    update.coefficients = function(p.values) {
      # Updates and returns p.values in place without copy. Also updates
      # self$coefficients in place.
      #
      # Args:
      #   p.values: R vector for predicted values (Gaussian model) with
      #     length as the number of observations.
      #
      # Returns:
      #   p.values: The same R vector (not a copy) with adjusted predictions.
      if (is.na(private$prediction.per.level) ||
          length(self$invvar.per.level) < length(self$coefficients)) return
      old.coefficients <- c(self$coefficients)  # make a copy
      residual.inv.variance <- private$residual.inv.var.callback()
      .UpdateCoefficients(private$updater, self$coefficients,
                          private$prediction.per.level,
                          self$invvar.per.level * residual.inv.variance)

      coefficient.changes <- self$coefficients - old.coefficients
      return(.UpdatePrediction(private$updater, p.values, coefficient.changes))
    },

    update.prior = function() {
      # Returns: NULL
      if (is.na(private$prediction.per.level) ||
          length(self$invvar.per.level) < length(self$coefficients)) return
      # TODO(kuehnelf): improve API for MH sampling prior in R code
      residual.inv.variance <- private$residual.inv.var.callback()
      .UpdateRanefPrior(private$updater,
                        self$coefficients,
                        private$prediction.per.level,
                        self$invvar.per.level * residual.inv.variance)
    },

    set.residual.inv.var.callback = function(callback) {
      private$residual.inv.var.callback <- callback
    }
  ),

  private = list(
    residual.inv.var.callback = NA
  )
)

ScaledGaussianRandomEffect <- R6Class("ScaledGaussianRandomEffect",
  inherit = GaussianRandomEffect,
  cloneable = FALSE,
  public = list(
    calc.immutable.stats = function(response, inverse.variance) {
      # For gaussian model, part of the immutable stats is the product
      # of response, offsets and scaling aggregated to the feature level.
      # This quantity is used in the collect.stats function.
      # Another quantity is the product of inverse variance with the square
      # or the scaling. This quantity is used in the both the update.prior
      # and update coefficients function.
      row2level.map <- .IndexerRowToLevelMap(.GetIndexReader(private$updater))
      scaling <- .IndexerRowScaling(.GetIndexReader(private$updater))
      # a negative index indicates that there a NA features
      idx <- (row2level.map < 0)
      row2level.map[idx] <- NA
      # this aggregates the product of response, inverse.variance and
      # scaling to the level id while leaving out NAs
      rivs <- response * inverse.variance * scaling
      self$cached.immutable.stats <- tapply(rivs, row2level.map, FUN = sum)
      invvar <- invvar * scaling * scaling
      # this is what we call invvar in the paper
      self$invvar.per.level <- tapply(invvar, row2level.map, FUN = sum)
      return(self$cached.immutable.stats)
    }
  )
)
