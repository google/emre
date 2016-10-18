# common shared utility function
.GenerateModelData <- function(
    nobs = 10000,
    group.sizes = c(500, 700, 1000),
    group.sds = c(0.1, 0.4, 0.5),
    bias = 60,
    feature.name = "x") {
  frm <- data.frame(y = rep(0, nobs), n = rep(bias, nobs), mean = 1)
  frm$n[1:as.integer(nobs / 3)] <- pmax(1, as.integer(bias / 10))

  group.vals <- list()
  for (j in 1:length(group.sizes)) {
    group.name <- paste(feature.name, j, sep = ".")
    group.vals[[group.name]] <- rgamma(group.sizes[j],
                                       shape = group.sds[j]^(-2),
                                       scale = group.sds[j]^2)
    grp.idx <- sample(1:group.sizes[j], nobs, replace = TRUE)
    stopifnot(group.sizes[j] == length(unique(grp.idx)))
    frm[[group.name]] <- as.factor(grp.idx)
    frm$mean <- frm$mean * group.vals[[group.name]][grp.idx]
  }
  frm$y <- rpois(nobs, frm$mean * frm$n)

  return(list(dat = frm, group.sds = group.sds, group.vals = group.vals,
              bias = bias))
}

.setUp <- function() {}

TestInitializeBiasRandomEffect <- function() {
  kNumObservations <- 999
  prior <- new(P("emre.FeatureFamilyPrior"))
  prior$model_class_type <- "POISSON"
  prior$ranef_update_type <- "OPTIMIZED"
  prior$mixture_components <- list(
      new(P("emre.PriorComponent"),
          gamma_alpha = 20, gamma_beta = 20))
  prior$inverse_variance <- 11.0
  prior$mean <- 1.0
  prior$feature_family <- "__bias__"

  # the index directory is ignored for the bias feature
  bias.indexer <- emre:::.CreateBiasIndexReader(kNumObservations)
  bias.ranef <- RandomEffect$new(prior, bias.indexer)

  checkEquals(bias.ranef$get.family.name(), "__bias__")
  checkEquals(bias.ranef$get.num.levels(), 1)
  checkEquals(length(bias.ranef$get.feature.levels()), 1)
  checkIdentical(bias.ranef$coefficients, c(1.0))

  # check the prior
  prior.out <- bias.ranef$get.prior()
  stopifnot(!is.null(prior))
  checkEquals(prior.out$model_class_type, 0)  # POISSON
  checkEquals(prior.out$ranef_update_type, 1)  # OPTIMIZED
  checkEquals(prior.out$prior_update_type, 2)  # INTEGRATED

  stopifnot(!is.null(prior.out$mixture_components))
  checkEquals(prior.out$inverse_variance, 11.0)
  checkEquals(prior.out$mean, 1.0)
}

TestRandomEffect <- function() {
  set.seed(15)
  kFeatureFamily <- "1__x.1"
  kGroupSize <- 75
  r <- .GenerateModelData(group.sizes = c(kGroupSize), group.sds = c(0.4))
  num.obs <- nrow(r$dat)

  prior <- new(P("emre.FeatureFamilyPrior"))
  prior$model_class_type <- "POISSON"
  prior$ranef_update_type <- "GIBBS_SAMPLED"
  prior$prior_update_type <- "GIBBS_INTEGRATED"
  prior$mixture_components <- list(
      new(P("emre.PriorComponent"),
          gamma_alpha = 20, gamma_beta = 20))
  prior$inverse_variance <- 20.0
  prior$mean <- 1.0
  prior$feature_family <- kFeatureFamily

  # construct indexer
  index.writer <- emre:::.CreateIndexWriter(kFeatureFamily)
  emre:::.IndexerWriteStringFeatures(index.writer, r$dat[["x.1"]])
  index.reader <- index.writer$close()

  # construct random effect
  ranef <- RandomEffect$new(prior, index.reader)
  checkEquals(ranef$get.family.name(), kFeatureFamily)
  checkEquals(ranef$get.num.levels(), kGroupSize)
  levels <- ranef$get.feature.levels()
  checkEquals(length(levels), kGroupSize)
  checkTrue(all(levels %in% unique(as.character(r$dat$x.1))))

  # test non-predictive behavior with uninformative coefficients
  p.events <- rep(1.0, num.obs)
  old.p.events <- c(p.events)  # make a copy
  p.events <- ranef$add.to.prediction(p.events)
  checkTrue(max(abs(old.p.events - p.events)) < 1e-5)

  # test predictive behavior with informative coefficients
  p.events <- rep(1.0, num.obs)
  old.p.events <- c(p.events)  # make a copy
  ranef$coefficients <- pmin(2, 1.01 + rexp(kGroupSize))
  p.events <- ranef$add.to.prediction(p.events)
  checkTrue(min(abs(old.p.events - p.events)) > 1e-6)
}

TestIsDoneOptimIterator <- function() {
  it <- OptimIterator$new(NULL, NULL, start.iter = 0, max.iter = 200)

  checkEquals(it$max.iter, 200)
  checkEquals(it$iter, 0)
  checkTrue(!it$is.done())

  it$set.max.iter(500)
  it$set.start.iter(501)
  checkEquals(it$max.iter, 500)
  checkEquals(it$iter, 501)
  checkTrue(it$is.done())
}

TestIteratorInSetupEMREoptim <- function() {
  # Fits a Gamma prior on simulated data.
  kFamilyNames <- c("__bias__", "1__x.1", "1__x.2", "1__x.3")
  set.seed(15)
  r <- .GenerateModelData()

  # Gibbs sample ranefs
  mdl <- SetupEMREoptim(
      "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + offset(n)",
      data = r$dat, model.constructor = PoissonEMRE)

  stopifnot(!is.null(mdl$optim.iterator))
  stopifnot(!is.null(mdl$setup$burnin))

  it <- mdl$optim.iterator  # short hand
  checkEquals(it$max.iter, 0)

  checkEquals(it$get.num.obs(), nrow(r$dat))
  checkEquals(it$get.num.levels("__bias__"), 1)
  checkTrue(!it$get.ranef("__bias__")$does.update.prior())
  checkEquals(it$get.num.levels("1__x.1"), 500)
  checkTrue(it$get.ranef("1__x.1")$does.update.prior())
  checkEquals(it$get.num.levels("1__x.2"), 700)
  checkTrue(it$get.ranef("1__x.2")$does.update.prior())
  checkEquals(it$get.num.levels("1__x.3"), 1000)
  checkTrue(it$get.ranef("1__x.3")$does.update.prior())

  x1.levels <- it$get.feature.levels("1__x.1")
  checkTrue(!is.null(x1.levels))
  checkEquals(length(x1.levels), it$get.num.levels("1__x.1"))
  checkTrue(all(x1.levels %in% unique(as.character(r$dat$x.1))))

  x2.levels <- it$get.feature.levels("1__x.2")
  checkTrue(!is.null(x2.levels))
  checkEquals(length(x2.levels), it$get.num.levels("1__x.2"))
  checkTrue(all(x2.levels %in% unique(as.character(r$dat$x.2))))

  x3.levels <- it$get.feature.levels("1__x.3")
  checkTrue(!is.null(x3.levels))
  checkEquals(length(x3.levels), it$get.num.levels("1__x.3"))
  checkTrue(all(x3.levels %in% unique(as.character(r$dat$x.3))))

  for (nm in kFamilyNames) {
    prior <- it$get.prior(nm)
    checkEquals(prior$model_class_type, 0)  # POISSON
    if (nm == "__bias__") {
      checkEquals(prior$ranef_update_type, 1)  # OPTIMIZED
      checkTrue(!it$get.ranef(nm)$does.update.prior())
    } else {
      checkEquals(prior$prior_update_type, 2)  # INTEGRATED (empirical bayes)
      checkEquals(prior$ranef_update_type, 0)  # GIBBS_SAMPLED
      checkTrue(it$get.ranef(nm)$does.update.prior())
    }
  }
}

TestUpdateModesInSetupEMREoptim <- function() {
  # Fits a Gamma prior on simulated data.
  kFamilyNames <- c("__bias__", "1__x.1", "1__x.2", "1__x.3")
  set.seed(15)
  r <- .GenerateModelData()

  # fully bayesian inference
  mdl1 <- SetupEMREoptim(
      "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + offset(n)",
      data = r$dat, model.constructor = PoissonEMRE, update.mode = "full.bayes")

  stopifnot(!is.null(mdl1$optim.iterator))

  it <- mdl1$optim.iterator  # short hand
  for (nm in kFamilyNames) {
    prior <- it$get.prior(nm)
    if (nm == "__bias__") {
      checkEquals(prior$ranef_update_type, 1)  # OPTIMIZED
      checkTrue(!it$get.ranef(nm)$does.update.prior())
    } else {
      checkEquals(prior$prior_update_type, 3)  # GIBBS_INTEGRATED (full bayes)
      checkEquals(prior$ranef_update_type, 0)  # GIBBS_SAMPLED
      checkTrue(it$get.ranef(nm)$does.update.prior())
    }
  }

  # fixed prior, gibbs sample ranefs
  mdl2 <- SetupEMREoptim(
    "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + offset(n)",
    data = r$dat, model.constructor = PoissonEMRE,
    update.mode = "fixed.prior.sample")

  stopifnot(!is.null(mdl2$optim.iterator))

  it <- mdl2$optim.iterator  # short hand
  for (nm in kFamilyNames) {
    prior <- it$get.prior(nm)
    if (nm == "__bias__") {
      checkEquals(prior$ranef_update_type, 1)  # OPTIMIZED
      checkTrue(!it$get.ranef(nm)$does.update.prior())
    } else {
      checkTrue(!it$get.ranef(nm)$does.update.prior())
      checkEquals(prior$ranef_update_type, 0)  # GIBBS_SAMPLED
    }
  }

  # fixed prior, optimize ranefs MAP, everything is a fixed effect!
  mdl3 <- SetupEMREoptim(
    "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + offset(n)",
    data = r$dat, model.constructor = PoissonEMRE,
    update.mode = "fixed.prior.map")

  stopifnot(!is.null(mdl3$optim.iterator))

  it <- mdl3$optim.iterator  # short hand
  for (nm in kFamilyNames) {
    prior <- it$get.prior(nm)
    checkTrue(!it$get.ranef(nm)$does.update.prior())
    checkEquals(prior$ranef_update_type, 1)  # OPTIMIZED
  }
}

TestFixedPriorMAPinInterleavedFit <- function() {
  # Fits Ranef coefficients without fitting a prior
  kFamilyNames <- c("__bias__", "1__x.1", "1__x.2", "1__x.3")
  set.seed(15)

  model1 <- .GenerateModelData(bias = 6000)
  model2 <- .GenerateModelData(group.sizes = c(1000, 500),
                               group.sds = c(0.6, 0.1),
                               feature.name = "w")

  mdl1 <- SetupEMREoptim(
      "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + offset(n)",
      data = model1$dat, model.constructor = PoissonEMRE,
      update.mode = "fixed.prior.map",
      thinning.interval = 10L)
  print("fitting gamma-poisson for model 1")
  mdl1 <- FitEMRE(mdl1, max.iter = 30, debug = TRUE)

  # this model must not interfere with model 1!
  mdl2 <- SetupEMREoptim(
      "y ~ 1 + (1|w.1) + (1|w.2) + offset(n)",
      data = model2$dat, model.constructor = PoissonEMRE,
      thinning.interval = 10L)
  print("fitting gamma-poisson for model 2")
  mdl2 <- FitEMRE(mdl2, max.iter = 30, debug = TRUE)

  # now interleave the fitting process
  for (max.iter in seq(40, 100, 10)) {
    print("continue fitting gamma-poisson for model 1")
    mdl1 <- FitEMRE(mdl1, max.iter = max.iter)

    print("continue fitting gamma-poisson for model 2")
    mdl2 <- FitEMRE(mdl2, max.iter = max.iter)
  }

  # we shouldn't have any llik trace
  checkTrue(is.null(mdl1$llik))
  checkTrue(is.null(mdl2$llik))

  # compare with a single run
  # TODO(kuehnelf): eliminate optim.iter from the settings
  mdl1.single.run <- SetupEMREoptim(
      "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + offset(n)",
      data = model1$dat, model.constructor = PoissonEMRE,
      update.mode = "fixed.prior.map")
  mdl1.single.run <- FitEMRE(mdl1.single.run, max.iter = 100,
                             thinning.interval = 10L, debug = FALSE)

  # check the snapshots schedules for model 1
  for (k in seq_along(kFamilyNames)) {
    nm <- kFamilyNames[k]
    ranef <- GetRanefs(mdl1, nm)
    ranef.sr <- GetRanefs(mdl1.single.run, nm)

    checkEquals(colnames(ranef), colnames(ranef.sr))
    checkEquals(rownames(ranef.sr), paste0(c(seq(20 + k - 1, 99, 10), 100)))
    checkEquals(rownames(ranef), paste0(c(20 + k - 1, seq(30, 100, 10))))
  }

  # check agreement between models
  kTol <- 1e-5
  for (nm in kFamilyNames) {
    print(sprintf("check model for %s", nm))
    fit1 <- GetRanefs(mdl1.single.run, nm, start.iter = 100, end.iter = 100)
    fit2 <- GetRanefs(mdl1, nm, start.iter = 100, end.iter = 100)

    rel.err <- abs(fit1 - fit2) / (abs(fit1) + abs(fit2))
    if (max(rel.err) > kTol) {
      print(sprintf("rel.err (max %.3e) too big for %d instances,",
                    max(rel.err), length(which(rel.err > kTol))))
    }
    checkTrue(max(rel.err) < kTol)
  }
}

TestEmpiricalBayesInInterleavedFit <- function() {
  # Fits a Gamma prior on simulated data.
  kFamilyNames <- c("__bias__", "x_1", "x_2", "x_3")
  set.seed(15)

  model1 <- .GenerateModelData()
  model2 <- .GenerateModelData(group.sizes = c(1000, 500),
                               group.sds = c(0.6, 0.1))

  mdl1 <- SetupEMREoptim(
      "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + offset(n)",
      data = model1$dat, model.constructor = PoissonEMRE,
      thinning.interval = 5L, debug = TRUE)
  print("fitting gamma-poisson for model 1")
  mdl1 <- FitEMRE(mdl1, max.iter = 30)

  mdl2 <- SetupEMREoptim(
      "y ~ 1 + (1|x.1) + (1|x.2) + offset(n)",
      data = model2$dat, model.constructor = PoissonEMRE,
      thinning.interval = 5L, debug = TRUE)
  print("fitting gamma-poisson for model 2")
  mdl2 <- FitEMRE(mdl2, max.iter = 30)

  for (max.iter in seq(40, 100, 10)) {
    print("continue fitting gamma-poisson for model 1")
    mdl1 <- FitEMRE(mdl1, max.iter = max.iter, thinning.interval = 3L)

    print("continue fitting gamma-poisson for model 2")
    mdl2 <- FitEMRE(mdl2, max.iter = max.iter, thinning.interval = 3L)
  }

  # compare with a single run
  mdl1.single.run <- SetupEMREoptim(
      "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + offset(n)",
      data = model1$dat, model.constructor = PoissonEMRE,
      thinning.interval = 10L)
  mdl1.single.run <- FitEMRE(mdl1.single.run, max.iter = 100,
                             llik.interval = 10, debug = FALSE)

  # check if llik trace exists
  checkEquals(length(mdl1.single.run$llik), 10)
  checkTrue(all(!is.na(mdl1.single.run$llik)))

  # only check model 1 priors
  kExpectedPriors <- list(
      x.1 = list(sd = c(0.05, 0.15)),
      x.2 = list(sd = c(0.35, 0.55)),
      x.3 = list(sd = c(0.4, 0.6)))

  # check model agreements
  kTol <- 1e-2
  for (nm in c("x.1", "x.2", "x.3")) {
    nm.mod <- paste0("1__", nm)
    x <- GetPrior(mdl1, nm.mod)[["100"]]
    mdl1.prior.sd <- x$inverse_variance^(-0.5)

    x <- GetPrior(mdl1.single.run, nm.mod)[["100"]]
    mdl1.sr.prior.sd <- x$inverse_variance^(-0.5)

    print(sprintf("%s: sd = %.3e, sd = %.3e",
                  nm, mdl1.prior.sd, mdl1.sr.prior.sd))
    rel.err <- abs(mdl1.prior.sd - mdl1.sr.prior.sd) /
               (mdl1.prior.sd + mdl1.sr.prior.sd)
    checkTrue(max(rel.err) < kTol)
    checkTrue(mdl1.prior.sd >= kExpectedPriors[[nm]]$sd[1])
    checkTrue(mdl1.prior.sd <= kExpectedPriors[[nm]]$sd[2])
  }
}
