TestRandomEffectFromFixefTerm <- function() {
  f <- FixefTerm$new("x.1")
  AddData(f, data.frame(x.1 = letters[rep(1:5, 3)]))
  response <- rep(1:5, 3)
  ranef <- ConstructRandomEffect(f, response = response)
  checkTrue(!is.null(ranef))
  checkEquals(ranef$get.family.name(), "x.1")
  checkEquals(ranef$get.num.levels(), 5)
  checkEquals(ranef$coefficients, rep(1.0, 5))
  checkEquals(as.numeric(ranef$events.per.level), 3 * 1:5)
  checkTrue(!ranef$does.update.prior())
}

TestRandomEffectFromOffsetTerm <- function() {
  o <- OffsetTerm$new("offset(n)")
  checkTrue(is.null(ConstructRandomEffect(o)))
}

TestRandomEffectFromInterceptTerm <- function() {
  i <- InterceptTerm$new("1")
  AddData(i, data.frame(response = seq(11:103)))
  ranef <- ConstructRandomEffect(i)
  checkTrue(!is.null(ranef))
  checkEquals(ranef$get.family.name(), "__bias__")
  checkEquals(ranef$get.num.levels(), 1)
  checkEquals(ranef$coefficients, c(1.0))
  checkEquals(ranef$events.per.level, c(sum(seq(11:103))))
  checkTrue(!ranef$does.update.prior())
}

TestRandomEffectFromRanefTerm <- function() {
  r <- RanefTerm$new("(1|x.1)")
  AddData(r, data.frame(x.1 = letters[rep(1:5, 3)]))
  response <- rep(1:5, 3)
  ranef <- ConstructRandomEffect(r, response = response)
  checkTrue(!is.null(ranef))
  checkEquals(ranef$get.family.name(), "1__x.1")
  checkEquals(ranef$get.num.levels(), 5)
  checkEquals(ranef$coefficients, rep(1.0, 5))
  checkEquals(as.numeric(ranef$events.per.level), 3 * 1:5)
  checkTrue(ranef$does.update.prior())
}
