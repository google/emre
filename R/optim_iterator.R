library(matrixStats)

OptimIterator <- R6Class("OptimIterator",
  public = list(
    iter = NA,
    max.iter = NA,
    p.events = c(),

    initialize = function(offset, start.iter = 0, max.iter = 100, ...) {
      private$offset <- offset
      self$iter <- start.iter
      self$max.iter <- max.iter
    },

    # public methods
    # these accessor functions are safe-guarded against failure
    get.feature.levels = function(ff.name) {
      if (!is.null(self$get.ranef(ff.name))) {
        return(self$get.ranef(ff.name)$get.feature.levels())
      }
    },
    get.num.levels = function(ff.name) {
      if (!is.null(self$get.ranef(ff.name))) {
        return(self$get.ranef(ff.name)$get.num.levels())
      }
    },
    get.prior = function(ff.name) {
      if (!is.null(self$get.ranef(ff.name))) {
        return(self$get.ranef(ff.name)$get.prior())
      }
    },

    get.feature.order = function() { names(private$ranef.families) },
    get.num.obs = function() { length(private$offset) },
    get.ranef = function(ff.name) { private$ranef.families[[paste0(ff.name)]] },
    set.start.iter = function(iter) {
      stopifnot(is.numeric(iter))
      self$iter <- iter
    },
    set.max.iter = function(iter) {
      stopifnot(is.numeric(iter))
      self$max.iter <- iter
    },

    is.done = function() { return(self$iter > self$max.iter) },

    # this function is chainable
    add.ranef = function(x) {
      stopifnot(inherits(x, "RandomEffect"))  # safe-guard
      new.ranef <- list(x)
      names(new.ranef)[1] <- x$get.family.name()
      private$ranef.families <- c(private$ranef.families, new.ranef)
      invisible(self)
    },

    snapshot = function(iter, k, trace) {
      # Method to keep track of fitting progress.
      #
      # Args:
      #   iter: iteration number
      #   k: family index
      #   trace: named list for prior and ranef snapshots,
      #     If trace$prior.snapshots[[family.name]][[iter]] is defined
      #     then the FeatureFamilyPrior proto will be added to this list.
      #     If a matrix trace$snapshots[[family.name]] with rowname iter
      #     is defined, then the ranef coefficients will inserted as a row.
      # Returns:
      #   updated trace
      if (is.null(trace)) return

      # take a prior snapshot
      if (!is.null(trace$prior.snapshots)) {
        trace <- self$snapshot.prior(iter, k, trace)
      }

      # take a ranef coefficient snapshot, snapshot[[feature.name]] is a matrix
      if (!is.null(trace$snapshots)) {
        trace <- self$snapshot.coefficients(iter, k, trace)
      }

      return(trace)
    },

    snapshot.coefficients = function(iter, k, trace) {
      for (nm in names(trace$snapshots)) {
        # find the families that need to be snapshotted at iteration
        if (paste0(iter) %in% rownames(trace$snapshots[[nm]])) {
          if (ncol(trace$snapshots[[nm]]) == 0) {
            # replace the placeholder with the appropriate matrix
            colnames <- self$get.feature.levels(nm)
            rownames <- rownames(trace$snapshots[[nm]])
            trace$snapshots[[nm]] <- matrix(nrow = length(rownames),
                                            ncol = length(colnames),
                                            dimnames = list(rownames, colnames))
          }
          trace$snapshots[[nm]][paste0(iter), ] <-
              self$get.ranef(nm)$coefficients
        }
      }
      return(trace)
    },

    snapshot.prior = function(iter, k, trace) {
      for (nm in names(trace$prior.snapshots)) {
        # find the families that need to be snapshotted at iteration
        if (!is.null(trace$prior.snapshots[[nm]][[paste0(iter)]])) {
          trace$prior.snapshots[[nm]][[paste0(iter)]] <- self$get.prior(nm)
        }
      }
      return(trace)
    },

    # before an iteration starts
    setup = function() {
      self$p.events <- c(private$offset)  # copy offset into p.events
      for (k in seq_along(private$ranef.families)) {
        # TODO(kuehnelf): why does this make a copy of p.events?
        self$p.events <-
            private$ranef.families[[k]]$add.to.prediction(self$p.events)
      }
    },

    # iterate over all feature families
    iterate = function(trace = NULL) {
      if (self$is.done()) {
        return(trace)
      }

      self$setup()
      for (k in seq_along(private$ranef.families)) {
        trace <- self$snapshot(self$iter, k, trace)
        self$process.family(self$iter, k)
      }
      self$iter <- self$iter + 1

      if (self$is.done()) {
        for (k in seq_along(private$ranef.families)) {
          trace <- self$snapshot(self$iter, k, trace)
        }
      }
      return(trace)
    },

    # these methods should be overridden for a custom update schedule
    update.prior = function(iter, k) {
      return(TRUE)
    },

    update.coefficients = function(iter, k) {
      return(TRUE)
    },

    # go through all steps to update for a single feature family
    process.family = function(iter, k) {
      ranef <- private$ranef.families[[k]]
      ranef$collect.stats(self$p.events, private$offset)
      if (self$update.prior(iter, k)) {
        ranef$update.prior()
      }
      if (self$update.coefficients(iter, k)) {
        ranef$update.coefficients(self$p.events)
      }
    }

  ),
  private = list(
    offset = c(),
    ranef.families = list()
  )
)

.GenerateSnapshotSchedule <- function(start.snapshot, max.iter, freq) {
  start.snapshot <- (floor(start.snapshot / freq) + 1) * freq

  if (start.snapshot > max.iter) {
    return(c())
  } else {
    return(seq(start.snapshot, max.iter, freq))
  }
}

.FitWithBasicOptimIterator <- function(mdl, start.iter, max.iter, ...) {
  # Fits the variances/regularization using the Monte Carlo EM algorithm (MCEM).
  #
  # Args:
  #   mdl: A model object
  # Returns:
  #   An updated model object with the fitted model
  stopifnot(!is.null(mdl$optim.iterator))
  it <- mdl$optim.iterator  # short hand

  # setup for the main iteration loop
  it$set.start.iter(start.iter)
  it$set.max.iter(max.iter)

  trace <- list(snapshots = list(), prior.snapshots = list())
  # set up the ranef & prior snapshot schedule
  feature.order <- it$get.feature.order()
  for (k in seq_along(feature.order)) {
    nm <- feature.order[k]
    # TODO(kuehnelf): make snapshots configurable for each effect
    freq <- mdl$setup$thinning.interval
    if (freq > 0) {
      start.snapshot <- max(start.iter, mdl$setup$burnin)
      rownames <- .GenerateSnapshotSchedule(start.snapshot, max.iter, freq)
      if (length(rownames) == 0) next
      # adjust the snapshot schedule by feature family index
      rownames <- pmin(rownames + (k - 1), max.iter)
      colnames <- it$get.feature.levels(nm)
      if (mdl$setup$debug) {
        cat(sprintf("feature %s: start-end snapshot (%d-%d), freq. %d\n",
                    nm, rownames[1], rownames[length(rownames)], freq))
      }
      # setting ncol with the number of levels
      # potentially pre-allocates alot of memory,
      # hence we only use a placeholder matrix here.
      trace[["snapshots"]][[nm]] <- matrix(nrow = length(rownames),
                                           ncol = 0,
                                           dimnames = list(rownames, NULL))
      prior.snapshots <- as.list(rep(NA, length(rownames)))
      names(prior.snapshots) <- rownames
      trace[["prior.snapshots"]][[nm]] <- prior.snapshots
    }
  }

  # a simple main iteration loop
  while (!it$is.done()) {
    if (mdl$setup$debug) cat(sprintf("iteration %d\n", it$iter))
    trace <- it$iterate(trace)
  }

  # concatenate prior snapshots
  mdl$prior.snapshots <- c(mdl$prior.snapshots, list())
  for (nm in names(trace$prior.snapshots)) {
    mdl$prior.snapshots[[nm]] <- c(mdl$prior.snapshots[[nm]],
                                   trace$prior.snapshots[[nm]])
  }

  # concatenate ranef snapshots
  mdl$snapshots <- c(mdl$snapshots, list())
  for (nm in names(trace$snapshots)) {
    mdl$snapshots[[nm]] <- rbind(mdl$snapshots[[nm]], trace$snapshots[[nm]])
  }

  # summarize fitted results
  mdl$fit <- list(last.iteration = it$iter)
  return(mdl)
}
