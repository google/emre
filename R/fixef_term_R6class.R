################################################################################
# The fixed effect R6 base class for poisson and gaussian models
################################################################################

FixefTerm <- R6Class("FixefTerm",
  inherit = BaseTerm,
  public = list(
    print = function(...) {
      cat("<FixefTerm> with\n", paste0(private$initial.prior),
          "numeric.level: ", paste0(private$numeric.level),  sep = "")
      invisible(self)
    },

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
                                       ranef.class = NULL) {
      EmreDebugPrint("Poisson FixTerm cref")
      if (missing(index.reader) || is.null(index.reader)) {
        stopifnot(!is.null(private$index.writer))
        index.reader <- private$index.writer$close()
      }
      if (is.null(ranef.class)) {
        if (private$numeric.level) {
          ranef.class <- ScaledRandomEffect
          private$initial.prior$model_class_type <- "SCALED_POISSON"
        } else {
          ranef.class <- RandomEffect
        }
      }
      EmreDebugPrint(ranef.class)

      return(super$construct.random.effect(index.reader,
                                           ranef.class = ranef.class))
    },

    recognize.term = function(formula.str) {
      # Fixef terms must not start with "("
      if (length(grep(x = formula.str, pattern = "^\\(.*", perl = TRUE))) {
        return(FALSE)
      }
      # Everything matches except the offset term, hence the FixefTerm
      # recognizer should be tried last!
      return(length(grep(x = formula.str, pattern = "^offset\\s*")) == 0)
    }
  ),

  private = list(
    init.prior = function() {
      super$init.prior()
      kStdDev <- 0.5
      private$initial.prior$mean <- 1.0
      private$initial.prior$inverse_variance <- 1 / kStdDev^2
      private$initial.prior$model_class_type <- "POISSON"
      private$initial.prior$prior_update_type <- "DONT_UPDATE"
      # fixed effects are not sampled
      private$initial.prior$ranef_update_type <- "OPTIMIZED"
    },

    create.data.frame = function(data) {
      coef.list <- sapply(
          self$parser$coef.terms,
          function(term) { eval(parse(text = term), envir = data) },
          simplify = FALSE)

      if (all(sapply(coef.list, is.numeric))) {
        coef <- Reduce("*", coef.list)
      } else {
        # we convert the list into function arguments
        coef <- do.call(paste, c(coef.list, sep = "*"))
      }

      # create data frame
      r <- data.frame(z = coef)
      nm <- private$initial.prior$feature_family  # short hand
      if (is.numeric(coef)) {
        private$numeric.level <- TRUE
        names(r)[1] <- paste0(nm, ".scaling")
        # TODO(kuehnelf): provide a custom indexer that doesn't need this column
        r[[nm]] <- as.factor(1)
      } else {
        if (private$numeric.level) {
          stop(paste0("Fixed effect '", nm, "' must have numeric values"))
        }
        names(r)[1] <- nm
      }

      return(r)
    },

    create.family.name = function() {
      idx <- order(self$parser$coef.terms)
      return(paste(self$parser$coef.terms[idx], collapse = "__"))
    },

    parse.term = function(formula.str, context) {
      super$parse.term(formula.str, context)
      str.elts <- ParseArgList(formula.str, start.at.leftparen = FALSE,
                               delim = ":", skip.whitespace = TRUE)
      left.of.colon <- str.elts$args[1]
      self$parser$coef.terms <- strsplit(left.of.colon, "*", fixed = TRUE)[[1]]
      if (length(str.elts$args) > 1) {
        right.of.colon <- ParseArgList(str.elts$args[[2]],
                                       start.at.leftparen = FALSE)
        self$parser$args <- ParseExtraArgs(right.of.colon$args, 0)
      }

      # construct a canonical family name
      private$initial.prior$feature_family <- private$create.family.name()
    },

    # instance variables
    numeric.level = FALSE,  # initially assume this feature is discrete
    index.writer = NULL
  )
)

################################################################################
# InterceptTerm
################################################################################

InterceptTerm <- R6Class("InterceptTerm",
  inherit = FixefTerm,
  public = list(
    # intercepts are implicit, we don't need to write them out.
    add.data = function(data) {
      stopifnot(!is.null(data$response))
      events <- sum(data[["response"]])
      private$num.observations <- private$num.observations + nrow(data)
      private$total.events <- private$total.events + events

      return(c(events))
    },

    construct.random.effect = function(index.reader,
                                       ranef.class = RandomEffect) {
      EmreDebugPrint("Poisson InterceptTerm cref")
      bias.indexer <- .CreateBiasIndexReader(private$num.observations)
      ranef <- super$construct.random.effect(bias.indexer,
                                             ranef.class = ranef.class)
      ranef$events.per.level <- c(private$total.events)

      return(ranef)
    },

    recognize.term = function(formula.str) {
      return(length(grep(x = formula.str, pattern = "^(0|1)")) > 0)
    }
  ),

  private = list(
    init.prior = function() {
      super$init.prior()
      private$initial.prior$feature_family <- "__bias__"
    },

    parse.term = function(formula.str, context) {
      # there are no parameters to the intercept term
    },

    num.observations = 0,
    total.events = 0
  )
)

################################################################################
# OffsetTerm(s)
################################################################################

OffsetTerm <- R6Class("OffsetTerm",
  inherit = FixefTerm,
  public = list(
    offset.vec = c(),

    initialize = function(formula.str = "offset(1)", context = NULL) {
      if (self$recognize.term(formula.str)) {
        private$parse.term(formula.str)
      }
    },

    add.data = function(data) {
      offset.data <- private$get.offset.data(data)
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
      return(length(grep(x = formula.str, pattern = "^offset\\(.*\\)$")) > 0)
    }
  ),

  private = list(
    get.offset.data = function(data) {
      stopifnot(!is.na(private$offset.term))
      return(eval(parse(text = private$offset.term), envir = data))
    },

    parse.term = function(formula.str, context) {
      self$parser$str <- formula.str
      private$offset.term <- gsub("^offset\\((.*)\\)$", "\\1",
                                  formula.str, perl = TRUE)
    },

    offset.term = NA
  )
)

LogOffsetTerm <- R6Class("LogOffsetTerm",
  inherit = OffsetTerm,
  public = list(
    recognize.term = function(formula.str) {
      return(length(grep(x = formula.str,
                         pattern = "^log.offset\\(.*\\)$")) > 0)
    }
  ),

  private = list(
    parse.term = function(formula.str, context) {
      self$parser$str <- formula.str
      private$offset.term <-
          paste0("log(", gsub("^log.offset\\((.*)\\)$", "\\1",
                              formula.str, perl = TRUE), ")")
    }
  )
)
