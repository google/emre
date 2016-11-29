OptimIterator <- R6Class("OptimIterator",
  public = list(
    iter = NA,
    max.iter = NA,
    p.events = c(),  # TODO(kuehnelf): rename to p.values (for gaussian model)

    # TODO(kuehnelf): maybe simplify API to pass in a named list
    initialize = function(response, offset, start.iter = 0,
                          max.iter = 100, ...) {
      private$response <- response
      private$offset <- offset
      self$iter <- start.iter
      self$max.iter <- max.iter
      EmreDebugPrint("OptimIterator initialized")
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

    is.done = function() { return(self$iter >= self$max.iter) },

    # this function is chainable
    add.ranef = function(x) {
      stopifnot(inherits(x, "RandomEffect"))  # safe-guard
      new.ranef <- list(x)
      names(new.ranef)[1] <- x$get.family.name()
      private$ranef.families <- c(private$ranef.families, new.ranef)
      invisible(self)
    },

    calc.llik = function() {
      if (length(self$p.events) > 0) {
        # computes poisson log likelihood up to a constant
        llik <- sum(private$response * log(self$p.events)) - sum(self$p.events)
        return(llik)
      } else {
        return(NA)
      }
    },

    snapshot = function(iter, trace) {
      # Method to keep track of fitting progress.
      #
      # Args:
      #   iter: iteration number
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
        trace <- self$snapshot.prior(iter, trace)
      }

      # take a ranef coefficient snapshot, snapshot[[feature.name]] is a matrix
      if (!is.null(trace$snapshots)) {
        trace <- self$snapshot.coefficients(iter, trace)
      }

      # trace llik as a way to diagnose convergence
      if (!is.null(trace$llik) && iter %in% names(trace$llik)) {
        trace$llik[[paste0(iter)]] <- self$calc.llik()
      }

      return(trace)
    },

    snapshot.coefficients = function(iter, trace) {
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

    snapshot.prior = function(iter, trace) {
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

    # after we iterated over all feature families
    finish = function() {
    },

    # iterate over all feature families
    iterate = function(trace = NULL) {
      if (self$is.done()) {
        return(trace)
      }

      if (self$iter == 0) {
        trace <- self$snapshot(self$iter, trace)
      }
      self$setup()
      for (k in seq_along(private$ranef.families)) {
        self$process.family(self$iter, k)
      }
      self$finish()
      self$iter <- self$iter + 1
      trace <- self$snapshot(self$iter, trace)

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
    offset = c(),  # unaggregated offset (Poisson model)
    response = c(),  # and response
    ranef.families = list()
  )
)

GaussOptimIterator <- R6Class("GaussOptimIterator",
  inherit = OptimIterator,
  cloneable = FALSE,
  public = list(
    residual.inv.var = list("0" = 1.0),  # by default assume 0 start iteration
    prior.residual.inv.var = list(mean = 1.0, sd = 0.6),

    initialize = function(..., context = NULL) {
      if (!is.null(context)) {
        if (context$update.mode == "full.bayes") {
          private$sample.variance <- TRUE
        } else if (context$update.mode == "empirical.bayes") {
          private$sample.variance <- FALSE
        } else {
          stop(paste("other update modes are not supported:",
                    context$update.mode))
        }
      }
      super$initialize(...)
    },

    # add the residual variance callback
    add.ranef = function(x) {
      stopifnot(inherits(x, "GaussianRandomEffect"))  # safe-guard
      x$set.residual.inv.var.callback(function() {
          return(as.numeric(tail(self$residual.inv.var, 1))) })
      super$add.ranef(x)
    },

    # we'll draw samples from the posterior for the residual variance
    finish = function() {
      # TODO(kuehnelf): change names p.events & offset to p.values & inv.var
      error.term <- private$response - self$p.events
      error.sqr <- sum(error.term * error.term * private$offset)
      M <- length(private$response)  # assumes unaggregated data
      p <- self$prior.residual.inv.var  # short hand
      eta <- p$mean / (p$sd * p$sd)
      theta <- p$mean * eta
      new.residual.inv.var <-
          ifelse(private$sample.variance,
                 rgamma(1, shape = theta + M * 0.5,
                        rate = eta + error.sqr * 0.5),
                 (theta + M * 0.5 - 1) / (eta + error.sqr * 0.5))  # mode
      EmreDebugPrint(sprintf("residual inv. variance %0.2f",
                             new.residual.inv.var))
      if (new.residual.inv.var <= 0) {
        stop(paste("residual variance is infinite,",
                   "cannot estimate residual variance."))
      }
      tmp <- list(val = new.residual.inv.var)
      names(tmp) <- paste0(self$iter + 1)  # +1 because this is called b. inc.
      self$residual.inv.var <- c(self$residual.inv.var, tmp)
    },

    calc.llik = function() {
      if (length(self$p.events) > 0) {
        # calculates gaussian llik up to a constant
        error.term <- (private$response - self$p.events)
        error.sqr <- sum(error.term * error.term * private$offset)
        llik <- (-0.5) * error.sqr * self$residual.inv.var
        llik <- llik - 0.5 * sum(log(private$offset))
        M <- length(private$response)  # assumes data has not been aggregated
        llik <- llik + 0.5 * M * sum(log(self$residual.inv.var))
        return(llik)
      } else {
        return(NA)
      }
    }
  ),

  private = list(
    sample.variance = TRUE
  )
)

.GenerateSnapshotSchedule <- function(start.snapshot, max.iter, freq) {
  #start.snapshot <- (floor(start.snapshot / freq) + 1) * freq
  if (start.snapshot > max.iter) {
    return(c())
  } else {
    return(unique(c(seq(start.snapshot, max.iter, freq), max.iter)))
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

  trace <- list(snapshots = list(), prior.snapshots = list(), llik = c())
  # set up the ranef & prior snapshot schedule
  feature.order <- it$get.feature.order()
  for (k in seq_along(feature.order)) {
    nm <- feature.order[k]
    # TODO(kuehnelf): make snapshots configurable for each effect
    freq <- mdl$setup$thinning.interval
    if (freq > 0) {
      if (mdl$setup$burnin > 0) {
        start.snapshot <- max(start.iter, mdl$setup$burnin + 1)
      }
      else {
        start.snapshot <- start.iter
      }
      rownames <- .GenerateSnapshotSchedule(start.snapshot, max.iter, freq)
      if (length(rownames) == 0) next
      # take snapshot for all feature families at the same iteration
      rownames <- unique(pmin(rownames, max.iter))
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
  # set up the llik trace schedule
  if (mdl$setup$llik.interval > 0) {
    colnames <- .GenerateSnapshotSchedule(start.iter, max.iter,
                                          mdl$setup$llik.interval)
    trace$llik <- rep(NA, length(colnames))
    names(trace$llik) <- colnames
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
                                   na.omit(trace$prior.snapshots[[nm]]))
  }

  # concatenate ranef snapshots
  mdl$snapshots <- c(mdl$snapshots, list())
  for (nm in names(trace$snapshots)) {
    mdl$snapshots[[nm]] <- rbind(mdl$snapshots[[nm]],
                                 na.omit(trace$snapshots[[nm]]))
  }

  # concatenate llik
  mdl$llik <- c(mdl$llik, na.omit(trace$llik))

  # summarize fitted results
  mdl$fit <- list(last.iteration = it$iter)
  return(mdl)
}
