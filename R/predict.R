# The functions declared in this file can serve as a basis
# of how to implement other predictor functions

PredictEMREmodelAverage <- function(mdl, test.data, ranef.families) {
  stopifnot(mdl$model.family == "poisson")
  # identify intercept term
  icpt.idx <- grep(x = names(mdl$predictors),
                   pattern = "^InterceptTerm\\.\\d+$")
  stopifnot(length(icpt.idx) == 1)
  # identify & process offset term
  offset.idx <- grep(x = names(mdl$predictors),
                     pattern = "^OffsetTerm\\.\\d+$")
  stopifnot(length(offset.idx) == 1)
  offset.term <- mdl$predictors[offset.idx]
  offset.data <- offset.term$get.offset.data(test.data)

  bias <- GetRanefs(mdl, "__bias__")
  pred <- matrix(nrow = nrow(bias), ncol = nrow(test.data))
  for (i in 1:nrow(bias)) {
    pred[i, ] <- offset.data * bias[i, 1]
    for (nm in ranef.families) {
      ranefs <- GetRanefs(mdl, paste0("1__", nm))
      coef <- ranefs[i, paste0(test.data[[nm]])]
      pred[i, ] <- pred[i, ] * coef
    }
  }
  # average the predictions
  avg.pred <- colMeans(pred)
  return(avg.pred)
}
