################################################################################
# The base R6 class for LMER formula term parsing and processing.
# All sub-classes have to override and implement the following functionality:
# 1.) recognize.term(formula.str): Returns TRUE if the LMER term is recognized
#    for this a sub-class.
# 2.) add.data(data.frame): Processes a (partial) data frame with
#    column name(s) that correspond to the random effect term name.
#    This method must allow to be called multiple times to process and add data
#    chunks.
# 3.) construct.random.effect: Create and return the configured RandomEffect R6
#    class (optim_iterator.R).

################################################################################
library(R6)

BaseTerm <- R6Class("BaseTerm",
  public = list(
    # Contains functionality that previously existed in the randeffect S3 class
    parser = list(),

    # TODO(kuehnelf): use active bindings for getters and setters
    get.initial.prior = function() {
      return(private$initial.prior)
    },

    set.initial.prior.from.string = function(value) {
      stopifnot(is.character(value))
      private$initial.prior <-
          RProtoBuf::readASCII(emre.FeatureFamilyPrior, value)
    },

    initialize = function(formula.str = NULL, context = NULL, ...) {
      if (!is.null(formula.str) && self$recognize.term(formula.str)) {
        private$init.prior()  # first initialize the prior
        private$parse.term(formula.str, context)  # then adjust the prior
      }
    },

    # intercepts are implicit, we don't need to write them out.
    add.data = function(data) {
      # Each term interprets data from a data frame, the specifics are to
      # be implemented in the sub-classes. Should throw an error if the
      # the data frame is not applicable for interpretation.
      #
      # Args:
      #   data: A data frame
      #
      # Returns:
      #   returns a processed data.frame or vector
    },

    construct.random.effect = function(index.reader,
                                       ranef.class = RandomEffect) {
      # Construct the R6 RandomEffect class for use in OptimIterator
      #
      # Args:
      #   index.reader: An S4 object with the index reader or NULL
      #   ranef.class: R6 RandomEffect class
      # Returns:
      #   R6 RandomEffect class or NULL
      EmreDebugPrint("BaseTerm cref")
      stopifnot(self$is.initialized())
      return(ranef.class$new(private$initial.prior, index.reader))
    },

    # specialize in sub-classes
    recognize.term = function(formula.str) { return(TRUE) },


    is.initialized = function() { return(!is.null(private$initial.prior)) }
  ),

  private = list(
    # Sets the initial.prior proto, specialize in sub-classes
    init.prior = function() {
      private$initial.prior <- new(RProtoBuf::P("emre.FeatureFamilyPrior"))
    },

    parse.term = function(formula.str, context) {
      # Parse a formula string of the form "(xxx|yyy)", "offset(n)", "1".
      # The details of parsing the term are subject to the sub-classes.
      #
      # Args:
      #   formula.str: the formula string
      #
      # Returns:
      #   returns itself, hence the function is chainable
      self$parser$str <- formula.str
    },

    initial.prior = NULL
  )
)

# Implements dynamic dispatch with R6 classes and sub-classes
ConstructRandomEffect <- function(x, response = NULL, offset = NULL,
                                  index.reader = NULL) {
  EmreDebugPrint(paste("entering ConstructRandomEffect", head(class(x), 1)))
  on.exit(EmreDebugPrint(paste("exiting ConstructRandomEffect",
                               head(class(x), 1))))
  ranef <- x$construct.random.effect(index.reader)
  if (!is.null(ranef) && !is.null(response)) {
    EmreDebugPrint("calculate immutable stats")
    ranef$calc.immutable.stats(response, offset)
  }
  return(ranef)
}

AddData <- function(x, data, ...) {
  EmreDebugPrint(paste("entering AddData", head(class(x), 1)))
  on.exit(EmreDebugPrint(paste("exiting AddData", head(class(x), 1))))
  x$add.data(data)
}

RecognizeTerm <- function(x, formula.str) {
  x$recognize.term(formula.str)
}
