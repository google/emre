.EvalRanef <- function(data, re.terms, fact.sep = "|*|") {
  # Evaluates the elements of 're.terms'. If any element is NA the result will
  # be NA.
  #
  # Args:
  #   data: A data frame
  #   re.terms: A list of expressions to evaluate
  #   fact.sep: A string separator used if length(re.terms) > 1
  # Returns:
  #   A vector of values.  If length(re.terms) > 1 these will be strings.  If
  #   the length is one, the type will depend on the expression in re.terms[[1]]
  fact <- eval(parse(text = re.terms[[1]]), envir = data)
  na.idx <- is.na(fact)
  if (length(re.terms) > 1) {
    for (k in 2:length(re.terms)) {
      fact2 <- eval(parse(text = re.terms[[k]]), envir = data)
      fact <- paste(fact, fact2, sep = fact.sep)
      na.idx <- na.idx | is.na(fact2)
    }
    fact[na.idx] <- NA
  }
  if (length(fact) == 1) {
    fact <- rep(fact, nrow(data))
  }
  return(fact)
}


.ModifyListWithTypeCoercion <- function(org.list, update.list) {
  # Merges an original list with new elements in the update list and
  # updates and type casts existing elements in the orignal list.
  #
  # Args:
  #   org.list: The original list
  #   update.list: The update list
  # Returns:
  #   A a single list with new and updated elements
  # Examples:
  #   'list(name="A", times=c(1.1,1.3)), list(name="B", num=3L) ->
  #            list(name="B", times=c(1.1,1.3), num=3L)'

  kCoercionFunc = list(
      double = as.double,
      integer = as.integer,
      logical = as.logical,
      character = as.character,
      list = as.list,
      vector = as.vector,
      closure = as.function)

  # coerce elements that are already in the original list
  coerced.nm <- c()
  keys <- intersect(names(org.list), names(update.list))
  for (nm in keys) {
    coercion.func <- kCoercionFunc[[typeof(org.list[[nm]])]]
    org.list[[nm]] <- coercion.func(update.list[[nm]])
    coerced.nm <- c(coerced.nm, nm)
  }

  # simply add new elements to the list
  keys <- setdiff(names(update.list), coerced.nm)
  if (length(keys) > 0) {
    org.list <- c(org.list, update.list[keys])
  }

  return(org.list)
}
