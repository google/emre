# Fits a GLM for a Poissonian response variable
# response ~ 1 + (1 | group1) + (1 | group2) + offset(log(n))
# using various hyper-paramter updating schemes.

VarianceUpdateExample <- function() {
  set.seed(15)
  GenerateData <- function() {
    ngroups1 <- 200
    ngroups2 <- 90000
    m <- 30000
    dat1 <- data.frame(
        group1 = sample(1:ngroups1, m, replace = TRUE),
        group2 = sample(1:ngroups1, m, replace = TRUE),
        offset = (1 + rpois(m, rexp(m, 0.01))))
    dat2 <- data.frame(
        group1 = sample(1:ngroups1, ngroups2, replace = TRUE),
        group2 = sample(ngroups1 + (1:ngroups2), ngroups2, replace = TRUE),
        offset = (1 + rpois(ngroups2, rexp(ngroups2, 1))))
    dat3 <- data.frame(
        group1 = sample(ngroups1 + (1:ngroups2), ngroups2, replace = TRUE),
        group2 = sample(1:ngroups1, ngroups2, replace = TRUE),
        offset = (1 + rpois(ngroups2, rexp(ngroups2, 1))))
    dat4 <- data.frame(
        group1 = sample(ngroups1 + (1:ngroups2), ngroups2, replace = TRUE),
        group2 = sample(ngroups1 + (1:ngroups2), ngroups2, replace = TRUE),
        offset = (1 + rpois(ngroups2, rexp(ngroups2, 1))))

    dat <- rbind(dat1, dat2, dat3, dat4)
    v1 <- rgamma(ngroups1 + ngroups2, 5, 5)
    v2 <- rgamma(ngroups1 + ngroups2, 0.6, 0.6)
    pois.rate <- v1[dat$group1] * v2[dat$group2] * 0.3 * dat$offset
    dat$response <- rpois(nrow(dat), pois.rate)

    v1.norm <- exp(rnorm(ngroups1 + ngroups2, 0, 5^(-0.5)))
    v2.norm <- exp(rnorm(ngroups1 + ngroups2, 0, 0.6^(-0.5)))
    pois.rate <- exp(v1.norm[dat$group1] + v2.norm[dat$group2]) * 0.3 *
        dat$offset
    dat$response.lmer <- rpois(nrow(dat), pois.rate)

    dat$group1 <- factor(dat$group1)
    dat$group2 <- factor(dat$group2)
    dat$n <- dat$offset
    return(dat)
  }

  dat <- GenerateData()
  if (sum(dat$response) != 1095019) {
    stop("Something is wrong with generating the test data!")
  }

  # fit model parameters with MCEM sampling
  cat("fit model with MCEM sampling\n")
  r.sample <- FitREGMH(
      "response ~ 1 + (1|group1) + (1|group2) + offset(n)",
      data = dat, model.family = "poisson", max.iter = 451,
      trace.freq = 450, prior.csv.trace.freq = 1, prior.optim.end = 400,
      prior.optim.type = "sample")

  # fit model parameters with MCEM Rao Blackwellized sampling
  cat("fit model with Rao-Blackwellized MCEM sampling\n")
  r.rao <- FitREGMH(
      "response ~ 1 + (1|group1) + (1|group2) + offset(n)",
      data = dat, model.family = "poisson", max.iter = 451,
      trace.freq = 450, prior.csv.trace.freq = 1, prior.optim.end = 400,
      prior.optim.type = "rao_blackwellized")

  # fit model parameters with MCEM Rao Blackwellized sampling
  cat("fit model with integrated MCEM sampling\n")
  r.integrated <- FitREGMH(
      "response ~ 1 + (1|group1) + (1|group2) + offset(n)",
      data = dat, model.family = "poisson", max.iter = 451,
      trace.freq = 450, prior.csv.trace.freq = 1, prior.optim.end = 400,
      prior.optim.type = "gibbs_integrated")
  return(r.integrated)
}
