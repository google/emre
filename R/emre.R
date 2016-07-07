# package to fit GLMs via Gibbs MCEM sampling
# To build & install this package type:
# blaze build -c opt contentads/analysis/caa/search_plus/regmh/emre:emre
# blaze run -c opt \
#  contentads/analysis/caa/search_plus/regmh/emre:emre_install
# in R
# library(emre)
# r <- VarianceUpdateExample()

PoissonEMRE <- function() {
  # The objects whose constructors are listed in 'feature.constructors' are
  # stored in predictors[[i]]$predictor
  r <- list(predictors = list(), response.formula = "",
            model.family = "poisson",
            # The following settings parameter are across all feature families.
            # Parameters for a given feature family are set in their respective
            # family classes, here "FixedEffect" and "RandEffect".
            # The parameters in these feature classes have precedence over
            # the ones below.
            setup = list(
                iterator.class = OptimIterator,
                start.iter = 0L,
                thinning.interval = 500L,  # intervals between taking samples
                burnin = 19L,
                # TODO(kuehnelf): set to full.bayes as a default
                update.mode = "empirical.bayes",
                max.iter = 0L,  # this will be set in FitEMRE
                llik.interval = 0L,  # non-zero value will calc llik per iter
                debug = FALSE),
            # TODO(kuehnelf): Terms in the formula string are recognized in
            # order of the elements in the feature.constructors list.
            # This means that the most generic term recognizer must come last,
            # however this makes it hard to add a new feature constructors.
            feature.constructors = list(
                InterceptTerm = InterceptTerm,
                RanefTerm = RanefTerm,
                ScaledRanefTerm = ScaledPoissonTerm,
                OffsetTerm = LogOffsetTerm,
                OffsetTerm = OffsetTerm,
                FixefTerm = FixefTerm))
  class(r) <- c("EMRE", "PoissonEMRE")
  return(r)
}

GaussianEMRE <- function() {
  EmreDebugPrint("calling GaussianEMRE")
  r <- PoissonEMRE()
  r$setup$iterator.class <- GaussOptimIterator
  r$model.family <- "gaussian"
  r$feature.constructors = list(
      InterceptTerm = GaussianInterceptTerm,
      RanefTerm = GaussianRanefTerm,
      ScaledRanefTerm = ScaledGaussianTerm,
      OffsetTerm = GaussianNoiseVarianceTerm,
      FixedEffect = GaussianFixefTerm)
  class(r) <- c(class(r), "GaussianEMRE")
  return(r)
}

.ConstructPredictors <- function(x, formula.str) {
  fc <- x$feature.constructors  # short hand
  constructors <- sapply(fc, function(z) z$new(), simplify = FALSE)
  str.elts <- ParseArgList(formula.str, start.at.leftparen = FALSE,
                           delim = c("+", "-"), skip.whitespace = TRUE,
                           left.paren = "(", right.paren = ")")
  r <- list()  # the empty predictor list

  for (j in seq_along(str.elts$args)) {
    predictor <- NULL
    term <- str.elts$args[[j]]
    for (k in seq_along(constructors)) {
      if (RecognizeTerm(constructors[[k]], term)) {
        predictor <- fc[[k]]$new(term, context = x$setup)
        r[[paste(names(constructors)[k], j, sep = ".")]] <- predictor
        break
      }
    }
    if (is.null(predictor)) {
      stop(paste("Did not recognize predictor:", term))
    }
  }

  # do some simple formula sanity checks to be consistent with LMER syntax:
  # 1.) add an intercept term if there is none.
  num.icpt <- length(grep(x = names(r), pattern = "^InterceptTerm\\.\\d+$"))
  if (num.icpt == 0) {
    EmreDebugPrint("adding implicit intercept term")
    r <- c(list(InterceptTerm.0 = fc$InterceptTerm$new("1")), r)
  } else if (num.icpt > 1) {
    stop(paste("Formula has more than one intercept term:", formula.str))
  }

  # 2.) add an intercept term if there is none.
  num.offset <- length(grep(x = names(r), pattern = "^OffsetTerm\\.\\d+$"))
  if (num.offset == 0) {
    EmreDebugPrint("adding implicit offset(1) term")
    r <- c(r, list(OffsetTerm.1000 = fc$OffsetTerm$new("offset(1)")))
  } else if (num.offset > 1) {
    stop(paste("Formula has more than one offset term:", formula.str))
  }

  x$predictors <- r
  return(x)
}

.SetupRegression <- function(mdl, formula.str, data = NULL, data.files = NULL,
                             data.reader.callback = NULL, ...) {
  # Does data pre-processing for the EMRE algorithm. Response and offset data
  # are stored in R as vectors. The feature family indexes map discrete feature
  # levels to the occurrence index in the response variable, and are stored
  # in memory via special IndexReader C++ classes.
  #
  # Args:
  #   mdl: a model of class EMRE
  #   formula.str: string giving the formula.  e.g. 'y ~ 1 + x1 + (1|group1)'
  #   data: data frame with columns y, x1 and group1
  #   data.files: list or vector of strings with files for input data that is
  #     used for model fitting. The data size can exceed available memory.
  #   data.reader.callback: callback function that takes a file name as an
  #     argument and returns a data frame. The default is 'read.table'
  stopifnot(inherits(mdl, "EMRE"))
  # handle the various data source cases
  loading.callback <- data.reader.callback
  if (!is.null(data)) {
    loading.callback <- function(...) { return(data) }
    data.files <- c("dataframe")
  } else if (is.null(data.reader.callback)) {
    loading.callback <- function(f) {
      return(read.table(f, header = TRUE, sep = ",", quote = "\""))
    }
  }
  stopifnot(!is.null(data.files), length(data.files) > 0)

  mdl$setup <- .ModifyListWithTypeCoercion(mdl$setup, list(...))
  mdl$formula.str <- FormulaToChar(formula.str)
  tmp <- strsplit(mdl$formula.str, "~")[[1]]
  mdl$response.formula <- Trim(tmp[[1]])
  mdl <- .ConstructPredictors(mdl, tmp[[2]])

  frm <- tryCatch(
      loading.callback(data.files[1]),
      error = function(cond) {
        stop("ERROR while reading -> abort\n")
      })

  # process data files for use with OptimIterator
  response <- c()
  for (j in seq_along(data.files)) {
    f <- data.files[j]
    cat(paste0("processing '", f, "'... "))

    df <- tryCatch(
        loading.callback(f),
        error = function(cond) {
          cat("ERROR while reading -> skip file\n")
          return(NA)
        })

    if (class(df) != "data.frame") {
      # skip this file if there was an error.
      if (!skip.read.errors) {
        stop("Failed to read file")
      }
      next
    } else if (is.null(df)) {
      cat("no data in file\n")
      next
    } else {
      cat(paste("done! Read", nrow(df), "lines.\n"))
    }

    # add the response variable
    df$response <- eval(parse(text = mdl$response.formula), envir = df)
    for (k in seq_along(mdl$predictors)) {
      tryCatch(
        AddData(mdl$predictors[[k]], df),
        error = function(cond) {
          stop(paste("The model cannot be used for this data frame:\n",
                     paste(names(df), collapse = ", "), "because of:\n",
                     cond))
        })
    }
    response <- c(response, df$response)
    df <- NULL
    gc()
  }

  # add the observation data to the mdl
  idx <- grep(x = names(mdl$predictors), pattern = "^OffsetTerm\\.\\d+$")
  stopifnot(length(idx) == 1)
  mdl$observation <- list(
      response = response,
      offset = mdl$predictors[[idx]]$offset.vec)

  return(mdl)
}

SetupEMREoptim <- function(formula.str, data = NULL, data.files = NULL,
                           data.reader.callback = NULL,
                           model.constructor = PoissonEMRE, ...) {
  # Fits the variances/regularization using the Monte Carlo EM algorithm (MCEM).
  #
  # Args:
  #   formula.str: string giving the formula.  e.g. 'y ~ 1 + x1 + (1|group1)'
  #   data: data frame with columns y, x1 and group1
  #   data.files: list or vector of strings with files for input data that is
  #     used for model fitting. The data size can exceed available memory.
  #   data.reader.callback: callback function that takes a file name as an
  #     argument and returns a data frame. The default is 'read.table'
  #   model.constructor: a sub-class of the model constructor, i.e.
  #     PoissonEMRE, GaussianEMRE,...
  #   max.iter: integer (default 190) to stop gibbs sampling at this iteration
  #   skip.read.errors: boolean (default FALSE). If TRUE, errors reading input
  #     data will be skipped and model fitting will continue.
  #   update.mode: A string with the following choices, 'full.bayes',
  #     'empirical.bayes', 'fixed.prior.sample' & 'fixed.prior.map'
  #   debug: boolean (default FALSE) print additional information along the way.
  # Returns:
  #   A list with elements:
  #     TODO
  # Examples:
  #   'y ~ 1 + x1 + (1|group1)
  #    y ~ 1 + x1 + (1|group1) + (1|group1:group2)
  #    y ~ 1 + x1 + (1|group.1, sd = 0.5)
  #
  #   data.files <- system(paste0("fileutil ls \"", data.file.pattern, "\""),
  #                        intern = TRUE)'
  # The general process to build the EMRE model:
  # 1.) GenerateInitalModel by parsing the formula string
  # 2.) Inspect the data and adjust the assumptions made to set up the
  #     initial model, i.e. discrete versus continuous features
  mdl <- model.constructor()
  stopifnot(inherits(mdl, "EMRE"))
  mdl <- .SetupRegression(mdl, formula.str, data = data,
                          data.files = data.files,
                          data.reader.callback = data.reader.callback, ...)
  stopifnot(!is.null(mdl$observation),
            !is.null(mdl$observation$offset),
            !is.null(mdl$observation$response))

  # set up a basic optim iterator
  mdl$optim.iterator <- mdl$setup$iterator.class$new(
      mdl$observation$response,
      mdl$observation$offset,
      max.iter = mdl$setup$max.iter,
      context = mdl$setup)  # TODO(kuehnelf): not very elegant

  # add random effects in the order they were parsed in the formula,
  # this is more transparent and simpler than a configuration flag, i.e.
  # optim.order = c(...)
  for (k in seq_along(mdl$predictors)) {
    ranef <- ConstructRandomEffect(mdl$predictors[[k]],
                                   response = mdl$observation$response,
                                   offset = mdl$observation$offset)
    if (!is.null(ranef)) {
      # add random effect to the gibbs sampler
      EmreDebugPrint(sprintf("add to iterator %s", class(ranef)))
      mdl$optim.iterator$add.ranef(ranef)
    }
  }
  # we don't need this anymore, clear memory for GC
  mdl$observation <- NULL
  mdl$prior.snapshots <- NULL
  mdl$snapshots <- NULL
  gc()

  return(mdl)
}

FitEMRE <- function(mdl, max.iter, ...) {
  # Fits the EMRE model for additional iterations. It may delete old snapshots
  # as it successfully runs.
  #
  # Args:
  #   mdl: A non-optional model object
  #   max.iter: A positive integer giving the number of iterations
  #   thinning.interval: A positive integer, or zero, for the number of
  #       iterations before taking a sample.
  #   update.mode: A string with the following choices, 'full.bayes',
  #     'empirical.bayes', 'fixed.prior.sample' & 'fixed.prior.map'
  #   llik.interval: A positive integer will compute the posterior
  #     log-likelihood, log P(response | ranefs, fixefs, offset, ...),
  #     at iterations spaced by llik.interval. Zero or negative
  #     will disable llik computations
  #   debug: boolean (default FALSE) print additional information along the way.
  # Returns:
  #   An updated model object

  start.iter <- mdl$setup$start.iter
  if (!is.null(mdl$fit) && is.numeric(mdl$fit$last.iteration)) {
    start.iter <- mdl$fit$last.iteration
  }
  stopifnot(start.iter >= 0)

  if (start.iter >= max.iter) {
    stop(sprintf("max.iter (%d) must be greater than start.iter (%d)",
                 max.iter, start.iter))
  }
  mdl$setup$max.iter <- max.iter
  mdl$setup <- .ModifyListWithTypeCoercion(mdl$setup, list(...))

  return(.FitWithBasicOptimIterator(mdl, start.iter, max.iter))
}
