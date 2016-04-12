TestScaledPoissonSetupOptim <- function() {
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


# TODO(kuehnelf): add more unit tests
