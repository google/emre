# These public utility functions provide access to the internal
# representation of estimated model parameters.

.FuzzyFeatureNameMatch <- function(available.names, family.name) {
  # Returns matched feature family names or NULL if not found
  #
  # Args:
  #   available.names: list or vector of strings with feature family names
  #   family.names: string of feature family to be matched
  # Returns:
  #   NULL or a string with matched feature family names

  # Let's clean up and canonicalize the family name
  selected.idx <- match(family.name, available.names)
  selected.name <- NULL
  if (is.na(selected.idx)) {
    clean.name <- gsub(x = family.name, pattern = "\\.", replacement ="_")
    potential.matches <- grep(x = available.names, pattern = clean.name,
                              ignore.case = TRUE)
    if (length(potential.matches) == 1) {
      selected.name <- available.names[potential.matches[1]]
      warning("the request didn't match exactly the available feature families")
    } else if (length(potential.matches) > 1) {
      warning(paste("several feature families matched the request:",
              paste0(available.names[potential.matches],
                     collapse = ", "),
                    sep = "\n"))
    } else {
      warning("this request doesn't match any feature family")
    }
  } else {
    # found an exact match
    selected.name <- available.names[selected.idx]
  }

  return(selected.name)
}

GetFamilyNames <- function(x) {
  # Returns a vector of feature family names used in the model.
  #
  # Args:
  #   x: an EMRE model object.
  #
  # Returns:
  #   a vector of strings with feature family names.
  stopifnot(inherits(x, "EMRE"), !is.null(x$optim.iterator))
  return(x$optim.iterator$get.feature.order())
}

GetRanefs <- function(x, family.name, start.iter = 0, end.iter = Inf,
                      max.levels = Inf) {
  # Returns a matrix of feature ranef coefficients (columns) for the requested
  # feature family. The dimension of the matrix is number of levels in this
  # feature family as number of columns and number of samples between
  # start.iter and end.iter as number of rows.
  #
  # Args:
  #   x: an EMRE model object
  #   family.name: a string with a feature family name
  #   start.iter: an integer for the start iteration
  #   end.iter: an integer for the highest iteration
  #   max.levels: an integer to limit the number of returned coefficients
  # Returns:
  #   a matrix with sampled ranef coefficients for feature levels (columns) and
  #   sample iteration (rows)
  stopifnot(!is.null(x), !is.null(x$snapshots))
  selected.name <- .FuzzyFeatureNameMatch(names(x$snapshots), family.name)
  stopifnot(!is.null(selected.name))

  iters <- as.numeric(rownames(x$snapshots[[selected.name]]))
  idx <- which(start.iter <= iters & iters <= end.iter)
  num.levels <- length(colnames(x$snapshots[[selected.name]]))
  num.selected.levels <- min(num.levels, max.levels)

  return(x$snapshots[[selected.name]][idx, 1:num.selected.levels, drop = F])
}

GetPrior <- function(x, family.name, start.iter = 0, end.iter = Inf) {
  # Returns a list of FeatureFamilyPrior protos for the feature family
  # Args:
  #   x: an EMRE model object.
  #   family.name: A string with the feature family
  #   start.iter: an integer for the start iteration
  #   end.iter: an integer for the highest iteration
  # Returns:
  #   a list (keys are snapshot iterations in increasing order)
  #   and FeatureFamilyPrior protos as values
  stopifnot(!is.null(x), !is.null(x$prior.snapshots))
  selected.name <- .FuzzyFeatureNameMatch(names(x$prior.snapshots),
                                          family.name)
  stopifnot(!is.null(selected.name))

  iters <- as.numeric(names(x$prior.snapshots[[selected.name]]))
  idx <- which(start.iter <= iters & iters <= end.iter)
  return(x$prior.snapshots[[selected.name]][idx])
}

GetResidualVariance <- function(x, start.iter = 0, end.iter = Inf) {
  # Returns a vector with residual variance samples for specified iterations
  # Args:
  #   x: a GaussianEMRE model object.
  # Returns:
  #   a double value with the residual variance estimate in the last iteration.
  stopifnot(inherits(x, "GaussianEMRE"), !is.null(x$optim.iterator))

  iters <- as.numeric(names(x$optim.iterator$residual.inv.var))
  idx <- which(start.iter <= iters & iters <= end.iter)
  return(1.0 / as.numeric(x$optim.iterator$residual.inv.var[idx]))
}
