NoTestScaledPoissonSetupOptim <- function() {
  bias <- 100.0
  is.expt <- 2.0
  set.seed(15)
  frm <- data.frame(is.expt = c(rep(-0.5, 100), rep(0.5, 100)))
  frm$bias <- bias
  frm$y <- rpois(nrow(frm), frm$bias * (is.expt ^ frm$is.expt))

  mdl <- SetupOptim(
      "y ~ 1 + is.expt",
      data = frm, model.constructor = PoissonEMRE,
      thinning.interval = 10)

  checkEquals(c("__bias__", "is.expt"), GetFamilyNames(mdl))
}

NoTestGLMmaximumLikelihoodFit <- function() {
  # fit a simple maximum likelihood model like glm does:
  # glm(y ~ 1 + is.expt, family = poisson)
  bias <- 100.0
  is.expt <- 2.0
  set.seed(15)
  frm <- data.frame(is.expt = c(rep(-0.5, 100), rep(0.5, 100)))
  frm$bias <- bias
  frm$n <- 1
  frm$y <- rpois(nrow(frm), frm$bias * (is.expt ^ frm$is.expt))

  print("maximum likelihood estimation, ala glm(y ~ 1 + is.expt)")
  mdl <- SetupOptim("y ~ 1 + is.expt + offset(log(n))",
      data = frm, model.constructor = PoissonEMRE,
      thinning.interval = 100L)

  mdl <- FitEMRE(mdl, max.iter = 20)

  est.bias <- GetRanefs(mdl, "__bias__")  # bias is on linear scale
  est.is.expt <- GetRanefs(mdl, "is.expt")  # this is on log scale
  checkEqualsNumeric(est.bias, bias, tol = 0.1)
  checkEqualsNumeric(exp(est.is.expt), is.expt, tol = 0.1)
}
