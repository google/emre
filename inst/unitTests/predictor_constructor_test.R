TestInterceptOffsetAndRandomEffectsTerm <- function() {
  mdl <- PoissonEMRE()
  mdl <- emre:::.ConstructPredictors(mdl, "1 + (1|w.2) + offset(1)")
  checkEquals(length(mdl$predictors), 3)

  checkEquals(names(mdl$predictors)[1], "InterceptTerm.1")
  prior <- mdl$predictors[[1]]$get.initial.prior()
  checkEquals(prior$feature_family, "__bias__")
  checkEquals(prior$model_class_type, 0)  # POISSON
  checkEquals(prior$prior_update_type, 999)  # DONT_UPDATE
  checkEquals(prior$ranef_update_type, 1)  # OPTIMIZED

  checkEquals(names(mdl$predictors)[2], "RanefTerm.2")
  prior <- mdl$predictors[[2]]$get.initial.prior()
  checkEquals(prior$feature_family, "1__w.2")
  checkEquals(prior$model_class_type, 0)  # POISSON
  checkEquals(prior$prior_update_type, 2)  # INTEGRATED
  checkEquals(prior$ranef_update_type, 0)  # GIBBS_SAMPLED

  checkEquals(names(mdl$predictors)[3], "OffsetTerm.3")
  # offset term must not construct a random effect
  checkTrue(is.null(emre:::ConstructRandomEffect(mdl$predictors[[3]])))
}

TestRandomEffectsOnlyTerm <- function() {
  mdl <- PoissonEMRE()
  mdl <- emre:::.ConstructPredictors(mdl, "(1|x.1)")
  checkEquals(length(mdl$predictors), 3)

  checkEquals(names(mdl$predictors)[1], "InterceptTerm.0")
  checkEquals(names(mdl$predictors)[2], "RanefTerm.1")
  prior <- mdl$predictors[[2]]$get.initial.prior()
  checkEquals(prior$feature_family, "1__x.1")
  checkEquals(prior$model_class_type, 0)  # POISSON
  checkEquals(prior$prior_update_type, 2)  # INTEGRATED
  checkEquals(prior$ranef_update_type, 0)  # GIBBS_SAMPLED

  checkEquals(names(mdl$predictors)[3], "OffsetTerm.1000")
  # offset term must not construct a random effect
  checkTrue(is.null(emre:::ConstructRandomEffect(mdl$predictors[[3]])))
}

TestInterceptOnlyTerm <- function() {
  mdl <- PoissonEMRE()
  mdl <- emre:::.ConstructPredictors(mdl, "1")
  checkEquals(length(mdl$predictors), 2)

  checkEquals(names(mdl$predictors)[1], "InterceptTerm.1")
  checkEquals(names(mdl$predictors)[2], "OffsetTerm.1000")
}

TestOffsetOnlyTerm <- function() {
  mdl <- PoissonEMRE()
  mdl <- emre:::.ConstructPredictors(mdl, "offset(1)")
  checkEquals(length(mdl$predictors), 2)

  checkEquals(names(mdl$predictors)[1], "InterceptTerm.0")
  checkEquals(names(mdl$predictors)[2], "OffsetTerm.1")
}
