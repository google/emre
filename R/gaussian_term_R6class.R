################################################################################
# The fixed effect R6 classes for gaussian models
################################################################################

GaussianFixefTerm <- R6Class("GaussianFixefTerm",
  inherit = FixefTerm,
  public = list(
    construct.random.effect = function(index.reader) {
      # Construct the R6 GaussianRandomEffect class for use in OptimIterator
      #
      # Args:
      #   index.reader: An S4 object with the index reader or NULL
      # Returns:
      #   R6 GaussianRandomEffect class or NULL
      EmreDebugPrint("Gaussian FixefTerm cref")
      if (private$numeric.level) {
        ranef.class <- ScaledGaussianRandomEffect
      } else {
        ranef.class <- GaussianRandomEffect
      }
      EmreDebugPrint(ranef.class)
      return(super$construct.random.effect(index.reader,
                                           ranef.class = ranef.class))
    }
  ),

  private = list(
    init.prior = function() {
      super$init.prior()
      kStdDev <- 5.0
      private$initial.prior$mean <- 0
      private$initial.prior$inverse_variance <- 1 / kStdDev^2
      private$initial.prior$model_class_type <- "GAUSSIAN"
      private$initial.prior$prior_update_type <- "DONT_UPDATE"
      # fixed effects are not sampled
      private$initial.prior$ranef_update_type <- "OPTIMIZED"
    }
  )
)

################################################################################
# InterceptTerm
################################################################################

GaussianInterceptTerm <- R6Class("GaussianInterceptTerm",
  inherit = InterceptTerm,  # perhaps more appropriate to derive from GaussFixef
  public = list(
    construct.random.effect = function(...) {
      EmreDebugPrint("Gaussian InterceptTerm cref")
      ranef <- super$construct.random.effect(bias.indexer,
                                             ranef.class = GaussianRandomEffect)
      return(ranef)
    }
  ),

  private = list(
    init.prior = function() {
      super$init.prior()
      kStdDev <- 5.0
      private$initial.prior$mean <- 0
      private$initial.prior$inverse_variance <- 1 / kStdDev^2
      private$initial.prior$model_class_type <- "GAUSSIAN"
    }
  )
)

################################################################################
# GaussianNoiseVarianceTerm
################################################################################

GaussianNoiseVarianceTerm <- R6Class("GaussianNoiseVarianceTerm",
  inherit = GaussianFixefTerm,
  public = list(
    offset.vec = c(),

    initialize = function(formula.str = "sd(0.1)", context = NULL) {
      if (self$recognize.term(formula.str)) {
        private$parse.term(formula.str)
      }
    },

    add.data = function(data) {
      offset.data <- private$get.stddev.data(data)
      if (length(offset.data) < nrow(data)) {
        # assuming 'data' is a data frame
        offset.data <- rep(offset.data, nrow(data))[1:nrow(data)]
      }
      self$offset.vec <- c(self$offset.vec, offset.data)
      return(offset.data)
    },

    # Offset term doesn't create a random effect
    construct.random.effect = function(...) {},

    recognize.term = function(formula.str) {
      return(length(grep(x = formula.str,
                         pattern = private$kGaussianNoiseVarianceRegex)) > 0)
    }
  ),

  private = list(
    parse.term = function(formula.str, ...) {
      private$offset.type <- gsub(
          x = formula.str,
          pattern = paste0("^", private$kGaussianNoiseVarianceRegex),
          replacement = "\\1",
          perl = TRUE)
      private$offset.term <- gsub(
          x = formula.str,
          pattern = paste0("^", private$kGaussianNoiseVarianceRegex),
          replacement = "\\2",
          perl = TRUE)
      if (!(private$offset.type %in% c("sd", "stddev"))) {
        stop(paste("unknown offset type: ", private$offset.type))
      }
      self$parser$str <- formula.str
    },

    get.stddev.data = function(data) {
      if (!is.na(private$offset.term)) {
        invvar <- eval(parse(text = private$offset.term), envir = data)^(-2)
        return(invvar)
      }
    },

    kGaussianNoiseVarianceRegex = "(sd|stddev)\\((.*)\\)$",
    offset.type = NA,
    offset.term = NA
  )
)

################################################################################
# The random effect R6 classes for gaussian models
################################################################################

GaussianRanefTerm <- R6Class("GaussianRanefTerm",
  inherit = RanefTerm,
  public = list(
    construct.random.effect = function(index.reader,
                                       ranef.class = GaussianRandomEffect) {
      # Construct the R6 GaussianRandomEffect class for use in OptimIterator
      #
      # Args:
      #   index.reader: An S4 object with the index reader or NULL
      # Returns:
      #   R6 GaussianRandomEffect class or NULL
      EmreDebugPrint("Gaussian RanefTerm cref")
      return(super$construct.random.effect(index.reader,
                                           ranef.class = ranef.class))
    }
  ),

  private = list(
    init.prior = function() {
      super$init.prior()
      kStdDev <- 0.2
      private$initial.prior$mean <- 0
      private$initial.prior$inverse_variance <- 1.0 / kStdDev^2
      private$initial.prior$model_class_type <- "GAUSSIAN"
      private$initial.prior$prior_update_type <- "INTEGRATED"
      private$initial.prior$ranef_update_type <- "GIBBS_SAMPLED"
    }
  )
)

ScaledGaussianTerm <- R6Class("ScaledGaussianTerm",
  inherit = ScaledPoissonTerm,
  public = list(
    construct.random.effect = function(index.reader) {
      # Construct the R6 GaussianRandomEffect class for use in OptimIterator
      #
      # Args:
      #   index.reader: An S4 object with the index reader or NULL
      # Returns:
      #   R6 GaussianRandomEffect class or NULL
      EmreDebugPrint("Scaled Gaussian RanefTerm cref")
      return(super$construct.random.effect(
          index.reader,
          ranef.class = ScaledGaussianRandomEffect))
    }
  ),

  private = list(
    init.prior = function() {
      super$init.prior()
      kStdDev <- 0.2
      private$initial.prior$mean <- 0
      private$initial.prior$inverse_variance <- 1.0 / kStdDev^2
      private$initial.prior$model_class_type <- "GAUSSIAN"
    }
  )
)

################################################################################


