TestFuzzyFeatureNameExactMatch <- function() {
  available.names <- c("x_1", "y_2", "z_1")
  res <- emre:::.FuzzyFeatureNameMatch(available.names, "x_1")
  checkEquals(res, "x_1")
}

TestFuzzyFeatureNameBroadMatch <- function() {
  available.names <- c("x_1", "y_2", "z_1")
  res <- emre:::.FuzzyFeatureNameMatch(available.names, "x.1")
  checkEquals(res, "x_1")

  res <- emre:::.FuzzyFeatureNameMatch(available.names, ".2")
  checkEquals(res, "y_2")
}

TestFuzzyFeatureNameNoMatch <- function() {
  available.names <- c("x_1", "y_2", "z_1")
  res <- emre:::.FuzzyFeatureNameMatch(available.names, "w_3")
  checkTrue(is.null(res))

  res <- emre:::.FuzzyFeatureNameMatch(available.names, "_1")
  checkTrue(is.null(res))
}

TestGetRanefs <- function() {
  row.names <- seq(5, 25, 5)
  col.names <- 1:20
  m <- matrix(nrow = length(row.names), ncol = length(col.names),
              dimnames = list(row.names, col.names))
  mdl <- list()
  mdl$snapshots <- list("1__x.1" = m)

  ranef1 <- GetRanefs(mdl, "1__x.1")
  checkEquals(rownames(ranef1), paste0(row.names))
  checkEquals(colnames(ranef1), paste0(col.names))

  ranef2 <- GetRanefs(mdl, "1__x.1", start.iter = 6, end.iter = 24)
  checkEquals(rownames(ranef2), paste0(seq(10, 20, 5)))
  checkEquals(colnames(ranef2), paste0(col.names))

  ranef3 <- GetRanefs(mdl, "1__x.1", start.iter = 6, end.iter = 24,
                           max.levels = 7)
  checkEquals(rownames(ranef3), paste0(seq(10, 20, 5)))
  checkEquals(colnames(ranef3), paste0(1:7))

  ranef4 <- GetRanefs(mdl, "1__x.1", start.iter = 11, end.iter = 15,
                           max.levels = 1)
  checkEquals(rownames(ranef4), "15")
  checkEquals(colnames(ranef4), "1")
}

TestGetPrior <- function() {
  all.priors <- list("10" = "a", "20" = "b", "30" = "c", "40"= "d")
  mdl <- list()
  mdl$prior.snapshots <- list("1__x.1" = all.priors)

  priors1 <- GetPrior(mdl, "1__x.1")
  checkEquals(names(priors1), names(all.priors))

  priors2 <- GetPrior(mdl, "1__x.1", start.iter = 11)
  checkEquals(names(priors2), names(all.priors)[2:4])

  priors3 <- GetPrior(mdl, "1__x.1", start.iter = 11, end.iter = 39)
  checkEquals(names(priors3), names(all.priors)[2:3])
}
