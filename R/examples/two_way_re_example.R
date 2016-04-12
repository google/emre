# A simple test case:
RegmhTwoWayReExample <- function(debug = TRUE, ...) {
  set.seed(15)
  GenerateTwoWayData <- function(
      nobs = 500,
      group.sizes = c(10, 20),
      group.sds = c(0.1, 0.4),
      bias = 6) {
    group.vals <- list()
    frm <- data.frame(y = rep(0, nobs), n = rep(1, nobs), mean = bias)
    for (j in 1:length(group.sizes)) {
      group.vals[[j]] <- rnorm(group.sizes[j], 0, group.sds[j])
      frm[[paste0("x.", j)]] <- sample(1:group.sizes[j], nobs, replace = TRUE)
      grp.idx <- frm[[paste0("x.", j)]]
      frm$mean <- frm$mean * exp(group.vals[[j]][grp.idx])
    }

    frm$y <- rpois(nobs, frm$mean * frm$n)
    frm$x.1 <- as.factor(frm$x.1)
    frm$x.2 <- as.factor(frm$x.2)

    return(list(dat = frm, group.sds = group.sds, group.vals = group.vals,
                bias = bias))
  }

  r <- GenerateTwoWayData()

  mdl <- FitREGMH(
      paste("y ~ 1 + offset(n) + (1|x.1,n.shards=3,prior.optim.freq=5)",
            "+ (1|x.2)"),
      data = r$dat, model.family = "poisson", max.iter = 1000,
      debug = debug, ...)

  return(mdl)
}


