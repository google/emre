################################################################################
# The base random effect R6 class for a discrete factor poisson model
################################################################################

RanefTerm <- R6Class("RanefTerm",
  inherit = BaseTerm,
  public = list(
    add.data = function(data) {
      if (self$is.initialized()) {
        df <- private$create.data.frame(data)
        nm <- private$initial.prior$feature_family  # short hand
        if (is.null(private$index.writer)) {
          private$index.writer <- .CreateIndexWriter(nm)
        }
        .IndexerWriteStringFeatures(private$index.writer,
                                    df[[nm]],
                                    scaling = df[[paste0(nm, ".scaling")]])
        return(df)
      }
    },

    construct.random.effect = function(index.reader,
                                       ranef.class = RandomEffect) {
      EmreDebugPrint("Poisson RanefTerm cref")
      if (missing(index.reader) || is.null(index.reader)) {
        stopifnot(!is.null(private$index.writer))
        index.reader <- private$index.writer$close()
      }

      return(super$construct.random.effect(index.reader,
                                           ranef.class = ranef.class))
    },

    recognize.term = function(formula.str) {
      if (length(grep(x = formula.str,
                      pattern = "^\\(\\s*1\\s*\\|\\s*1\\s*(,|\\))",
                      perl = TRUE)) > 0) {
        # exclude terms like (1 | 1, ...)
        return(FALSE)
      }

      return(length(grep(x = formula.str,
                         pattern = "^\\(\\s*1\\s*\\|.*",
                         perl = TRUE)) > 0)
    }
  ),

  private = list(
    init.prior = function() {
      super$init.prior()
      private$initial.prior$mean <- 1.0
      private$initial.prior$inverse_variance <- 20.0
      private$initial.prior$model_class_type <- "POISSON"
      # default use EM step prior update
      private$initial.prior$prior_update_type <- "INTEGRATED"
      # initially, we assume sampling from the posterior distribution
      private$initial.prior$ranef_update_type <- "GIBBS_SAMPLED"
    },

    create.data.frame = function(data) {
      # Create a data frame from data for the intercept random effect predictor,
      # i.e. (1|campaign.id)
      #
      # Args:
      #   data: A data frame
      # Returns:
      #   A data frame with columns corresponding to the predictor
      re.fact <- .EvalRanef(data, self$parser$re.terms, fact.sep = "|*|")
      r <- data.frame(z = as.factor(re.fact))
      names(r)[1] <- private$initial.prior$feature_family
      return(r)
    },

    create.family.name = function() {
      # Constructs canonical feature names from the RE-formuls.
      #
      # Returns:
      #   A a single string with the canonicalized feature name
      #   family name.
      # Examples:
      #   '(1|group.1) -> group.1'
      #   '(1|group1:group2) -> group1__group2'
      idx <- order(self$parser$re.terms)
      re.fam <- paste(self$parser$re.terms[idx], collapse = "__")

      idx <- order(self$parser$coef.terms)
      coef.fam <- paste(self$parser$coef.terms[idx], collapse = "__")

      return(paste(coef.fam, re.fam, sep = "__"))
    },

    process.prior.args = function(args) {
      EmreDebugPrint(paste("exiting process.prior.args:",
                     sprintf("%s=%s", names(args), args)))
      if ("sd" %in% names(args)) {
        private$initial.prior$inverse_variance <- as.double(args)^(-2)
      }
      if ("var" %in% names(args)) {
        private$initial.prior$inverse_variance <- as.double(args)^(-1)
      }
    },

    parse.term = function(formula.str, context = NULL) {
      # Parse a formula string of the form "(xxx|yyy)". The input string can
      # contain trailing terms. e.g. if "(xxx|yyy) + z", the " + z" will be
      # ignored.
      super$parse.term(formula.str)
      str.elts <- ParseArgList(formula.str, start.at.leftparen = TRUE,
                               delim = "|", skip.whitespace = TRUE)
      stopifnot(length(str.elts$args) == 2)

      right.of.bar <- ParseArgList(str.elts$args[[2]],
                                   start.at.leftparen = FALSE)
      stopifnot(length(right.of.bar$args) >= 1)

      r <- ParseArgList(str.elts$args[[1]], start.at.leftparen = FALSE,
                        delim = c("+", "-"), skip.whitespace = TRUE,
                        left.paren = "(", right.paren = ")")
      self$parser$coef.terms <- r$args
      self$parser$re.terms <- ParseRanef(right.of.bar$args[[1]])
      self$parser$args <- ParseExtraArgs(right.of.bar$args, 1)

      # remove the no intercept term, i.e. (0 + x | id)
      idx <- match("0", self$parser$coef.terms)
      if (!is.na(idx)) {
        self$parser$coef.terms <- self$parser$coef.terms[-idx]
      }

      # guard against an implicit or explicit intercept terms, i.e.
      # (x | id), (1 + x | id)
      idx <- match("1", self$parser$coef.terms)
      if (is.na(idx)) {
        # we have an implicit intercept & coefficient terms
        stop("Implicit intercept term, (x | ...), not supported")
      } else if (length(self$parser$coef.terms) > 1) {
        # we have both an explicit intercept term & coefficient term
        stop(paste("Intercept with coefficient term, (1 + x | ...),",
                   "is not supported"))
      }

      # construct a canonical family name
      private$initial.prior$feature_family <- private$create.family.name()

      # preset other parameters for this prior
      private$process.prior.args(self$parser$args)

      # set the global choice for the update.mode
      if (!is.null(context)) {
        if (context$update.mode == "full.bayes") {
          EmreDebugPrint(paste0("gibbs sample prior and ranef coeff. for, ",
                                private$initial.prior$feature_family))
          private$initial.prior$prior_update_type <- "GIBBS_INTEGRATED"
          private$initial.prior$ranef_update_type <- "GIBBS_SAMPLED"
        } else if (context$update.mode == "empirical.bayes") {
          EmreDebugPrint(paste0("optimize prior and sample ranef coeff. for, ",
                                private$initial.prior$feature_family))
          private$initial.prior$prior_update_type <- "INTEGRATED"
          private$initial.prior$ranef_update_type <- "GIBBS_SAMPLED"
        } else if (context$update.mode == "fixed.prior.sample") {
          EmreDebugPrint(paste0("fixed prior and sample ranef coeff. for, ",
                                private$initial.prior$feature_family))
          private$initial.prior$prior_update_type <- "DONT_UPDATE"
          private$initial.prior$ranef_update_type <- "GIBBS_SAMPLED"
        } else {
          EmreDebugPrint(paste0("fixed prior and optimize ranef coeff. for, ",
                      private$initial.prior$feature_family))
          private$initial.prior$prior_update_type <- "DONT_UPDATE"
          private$initial.prior$ranef_update_type <- "OPTIMIZED"
        }
      }
    },

    # instance variables
    index.writer = NULL
  )
)

################################################################################
# ScaledPoissonTerm
################################################################################

ScaledPoissonTerm <- R6Class("ScaledPoissonTerm",
  inherit = RanefTerm,
  public = list(
    construct.random.effect = function(index.reader) {
      EmreDebugPrint("Scaled Poisson RanefTerm cref")
      return(super$construct.random.effect(index.reader,
                                           ranef.class = ScaledRandomEffect))
    },

    recognize.term = function(formula.str) {
      if (length(grep(x = formula.str, pattern = "^\\(\\s*1\\s*\\|.*")) > 0) {
        return(FALSE)
      } else {
        # this matches terms like (x.1 | ...
        return(length(grep(x = formula.str, pattern = "^\\([^\\|]*\\|.*")) > 0)
      }
    }
  ),

  private = list(
    init.prior = function() {
      super$init.prior()
      private$initial.prior$default_score <- 0.0
      private$initial.prior$mean <- 0.0
      private$initial.prior$inverse_variance <- 20.0
      private$initial.prior$model_class_type <- "SCALED_POISSON"
      private$initial.prior$prior_update_type <- "SAMPLE_FOR_SCALED"
      # initially, we assume sampling from the posterior distribution
      private$initial.prior$ranef_update_type <- "GIBBS_SAMPLED"
    },

    create.data.frame = function(data) {
      # Create a data frame from data for the intercept randeffect predictor,
      # i.e. (is.expt|campaign.id)
      #
      # Args:
      #   data: A data frame
      # Returns:
      #   A data frame with columns corresponding to the predictor
      column.name <- private$initial.pior$feature_family
      # we must have a single numeric coefficient term
      stopifnot(length(self$parser$coef.terms) == 1,
                self$parser$coef.terms != "1")
      term <- self$parser$coef.terms[1]
      coef <- eval(parse(text = term), envir = data)
      # we must have a continuous coefficient
      stopifnot(is.double(coef))
      r <- data.frame(z = coef)
      names(r)[1] <- paste0(column.name, ".scaling")

      if (length(self$parser$re.terms) == 1 && self$parser$re.terms[1] == "1") {
        # TODO(kuehnelf): this is sort of a hack to use
        # the current indexer infrastructure for a single
        # level feature with scaling, i.e (x|1)
        r[[column.name]] <- as.factor(1)
      } else {
        re.fact <- EvalRanef(data, self$parser$re.terms, fact.sep = "|*|")
        r[[column.name]] <- as.factor(re.fact)
      }

      return(r)
    }
  )
)

