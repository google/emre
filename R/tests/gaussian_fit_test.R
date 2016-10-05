# common shared utility function
.GenerateGaussianModelData <- function(
    nobs = 10000,
    group.sizes = c(500, 700, 1000),
    group.sds = c(0.1, 0.4, 0.5),
    noise.sd = 0.03,
    residual.variance = 1.0,
    feature.name = "x") {
  frm <- data.frame(y = rep(0, nobs), mean = 1.2)
  group.vals <- list()
  for (j in 1:length(group.sizes)) {
    group.name <- paste(feature.name, j, sep = ".")
    group.vals[[group.name]] <- rnorm(group.sizes[j], sd = group.sds[j])
    grp.idx <- sample(1:group.sizes[j], nobs, replace = TRUE)
    frm[[group.name]] <- as.factor(grp.idx)
    frm$mean <- frm$mean + group.vals[[group.name]][grp.idx]
  }

  frm$noise.sd <- noise.sd
  frm$y <- rnorm(nobs, frm$mean, sd = frm$noise.sd * sqrt(residual.variance))

  return(list(dat = frm, group.sds = group.sds, group.vals = group.vals,
              noise.sd = noise.sd))
}

.setUp <- function() {}

TestGaussianRandomEffect <- function() {
  set.seed(15)
  kFeatureFamily <- "1__x.1"
  kGroupSize <- 75
  r <- .GenerateGaussianModelData(group.sizes = c(kGroupSize),
                                  group.sds = c(0.4))
  num.obs <- nrow(r$dat)

  prior <- new(P("emre.FeatureFamilyPrior"))
  prior$model_class_type <- "GAUSSIAN"
  prior$ranef_update_type <- "GIBBS_SAMPLED"
  prior$prior_update_type <- "INTEGRATED"
  prior$mixture_components <- list(
      new(P("emre.PriorComponent"),
          mean = 0.0, inverse_variance = 20))
  prior$inverse_variance <- 20.0
  prior$mean <- 0.0
  prior$feature_family <- kFeatureFamily

  # construct indexer
  index.writer <- .CreateIndexWriter(kFeatureFamily)
  .IndexerWriteStringFeatures(index.writer, r$dat[["x.1"]])
  index.reader <- index.writer$close()

  # construct gaussian random effect
  ranef <- GaussianRandomEffect$new(prior, index.reader)
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

TestGaussianSetupEMREoptim <- function() {
  kFamilyNames <- c("__bias__", "1__x.1", "1__x.2", "1__x.3")
  set.seed(15)
  r <- .GenerateGaussianModelData()

  mdl <- SetupEMREoptim(
      "y ~ 1 + (1|x.1, sd = 0.5) + (1|x.2) + (1|x.3) + stddev(noise.sd)",
      data = r$dat, model.constructor = GaussianEMRE)

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
    checkEquals(prior$model_class_type, 1)  # GAUSSIAN
    if (nm == "__bias__") {
      checkEquals(prior$ranef_update_type, 1)  # OPTIMIZED
      checkEquals(prior$prior_update_type, 999)  # DONT_UPDATE
      checkTrue(!it$get.ranef(nm)$does.update.prior())
    } else {
      checkEquals(prior$ranef_update_type, 0)  # GIBBS_SAMPLED
      checkEquals(prior$prior_update_type, 2)  # INTEGRATED
      checkTrue(it$get.ranef(nm)$does.update.prior())
    }
  }
}

TestFitGaussianFullBayes <- function() {
  set.seed(15)
  r <- .GenerateGaussianModelData()

  mdl <- SetupEMREoptim(
    "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + stddev(noise.sd)",
    data = r$dat, model.constructor = GaussianEMRE,
    update.mode = "full.bayes", burnin = 20, thinning.interval = 5)

  # MCMC sample the prior variances
  print("fitting gauss-gauss model: full bayes")
  mdl <- FitEMRE(mdl, max.iter = 100, debug = TRUE)

  kExpectedPriors <- list(
      x.1 = c(0.05, 0.15),
      x.2 = c(0.35, 0.55),
      x.3 = c(0.4, 0.6))

  kExpectedPriorStdDevs <- list(
    x.1 = c(0.001, 0.006),
    x.2 = c(0.001, 0.014),
    x.3 = c(0.005, 0.02))

  for (nm in c("x.1", "x.2", "x.3")) {
    ranef.nm <- paste0("1__", nm)
    x.prior <- sapply(GetPrior(mdl, ranef.nm),
                      function(z) { z$inverse_variance^(-0.5) })
    x.prior <- as.vector(x.prior)

    checkTrue(mean(x.prior) >= kExpectedPriors[[nm]][1])
    checkTrue(mean(x.prior) <= kExpectedPriors[[nm]][2])
    checkTrue(sd(x.prior) >= kExpectedPriorStdDevs[[nm]][1])
    checkTrue(sd(x.prior) <= kExpectedPriorStdDevs[[nm]][2])
  }
}

# TODO(kuehnelf): add a scaled version to this test
TestFitGaussianEmpiricalBayes <- function() {
  set.seed(15)
  r <- .GenerateGaussianModelData()

  mdl <- SetupEMREoptim(
    "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + stddev(noise.sd)",
    data = r$dat, model.constructor = GaussianEMRE,
    update.mode = "empirical.bayes", burnin = 20, thinning.interval = 5)

  # Fit the prior variance parameters using EM
  print("fitting gauss-gauss model: empirical bayes")
  mdl <- FitEMRE(mdl, max.iter = 100, debug = TRUE)

  kExpectedPriors <- list(
      x.1 = c(0.05, 0.15),
      x.2 = c(0.35, 0.55),
      x.3 = c(0.4, 0.6))

  for (nm in c("x.1", "x.2", "x.3")) {
    ranef.nm <- paste0("1__", nm)
    x.prior <- sapply(GetPrior(mdl, ranef.nm),
                      function(z) { z$inverse_variance^(-0.5) })
    x.prior <- as.vector(x.prior)

    checkTrue(tail(x.prior, 1) >= kExpectedPriors[[nm]][1])
    checkTrue(tail(x.prior, 1) <= kExpectedPriors[[nm]][2])

  }
}

TestFitGaussianFullBayesResidualVariance <- function() {
  set.seed(15)
  r <- .GenerateGaussianModelData(residual.variance = 4.0)

  mdl <- SetupEMREoptim(
    "y ~ 1 + (1|x.1) + (1|x.2) + (1|x.3) + stddev(noise.sd)",
    data = r$dat, model.constructor = GaussianEMRE,
    update.mode = "full.bayes", burnin = 20, thinning.interval = 5)

  # MCMC sample the prior variances
  print("fitting gauss-gauss model: full bayes")
  mdl <- FitEMRE(mdl, max.iter = 100, debug = TRUE)

  # check estimated residual variance
  res.var <- tail(GetResidualVariance(mdl), 1)
  checkTrue(res.var >= 3.90)
  checkTrue(res.var <= 4.10)

  kExpectedPriors <- list(
      x.1 = c(0.05, 0.15),
      x.2 = c(0.35, 0.55),
      x.3 = c(0.4, 0.6))

  kExpectedPriorStdDevs <- list(
    x.1 = c(0.001, 0.02),
    x.2 = c(0.001, 0.015),
    x.3 = c(0.005, 0.015))

  for (nm in c("x.1", "x.2", "x.3")) {
    ranef.nm <- paste0("1__", nm)
    x.prior <- sapply(GetPrior(mdl, ranef.nm),
                      function(z) { z$inverse_variance^(-0.5) })
    x.prior <- as.vector(x.prior)

    checkTrue(mean(x.prior) >= kExpectedPriors[[nm]][1])
    checkTrue(mean(x.prior) <= kExpectedPriors[[nm]][2])
    checkTrue(sd(x.prior) >= kExpectedPriorStdDevs[[nm]][1])
    checkTrue(sd(x.prior) <= kExpectedPriorStdDevs[[nm]][2])
  }
}
