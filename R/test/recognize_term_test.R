# All functions matching the regular expression "^Test" are interpreted and run
# as unit tests by default. You can change this with a parameter to
# RunUnitTests().

TestFixEfTerm <- function() {
  f <- FixefTerm$new()
  checkTrue(inherits(f, "FixefTerm"))
  checkTrue(RecognizeTerm(f, "x"))
  checkTrue(!RecognizeTerm(f, "(x"))
  checkTrue(!RecognizeTerm(f, "offset(n)"))
  checkTrue(!RecognizeTerm(f, "offset( log(n))"))
}

TestOffsetTerm <- function() {
  o <- OffsetTerm$new()
  checkTrue(inherits(o, "OffsetTerm"))
  checkTrue(RecognizeTerm(o, "offset(n)"))
  checkTrue(RecognizeTerm(o, "offset( log(n))"))
  checkTrue(RecognizeTerm(o, "offset( log(n) )"))
  checkTrue(RecognizeTerm(o, "offset(1)"))
  # unfortunately, regex doesn't deal with unbalanced paranthesis
  checkTrue(RecognizeTerm(o, "offset( log(n)"))
  # checkFalse would be more appropriate
  checkTrue(!RecognizeTerm(o, "offset"))
  checkTrue(!RecognizeTerm(o, "offset (n)"))
  checkTrue(!RecognizeTerm(o, "offset( n"))
}

TestInterceptTerm <- function() {
  i <- InterceptTerm$new()
  checkTrue(inherits(i, "InterceptTerm"))
  checkTrue(RecognizeTerm(i, "0"))
  checkTrue(RecognizeTerm(i, "1"))
  # checkFalse would be more appropriate
  checkTrue(!RecognizeTerm(i, "2"))
  checkTrue(!RecognizeTerm(i, "(1|x)"))
  checkTrue(!RecognizeTerm(i, "(1)"))
}

TestRandEffect <- function() {
  r <- RanefTerm$new()
  checkTrue(inherits(r, "RanefTerm"))
  checkTrue(RecognizeTerm(r, "(1|x)"))
  checkTrue(RecognizeTerm(r, "(1|x, sd = 0.5)"))
  checkTrue(!RecognizeTerm(r, "(y|x)"))
  checkTrue(!RecognizeTerm(r, "(1|1)"))
  checkTrue(!RecognizeTerm(r, "(1| 1)"))
  checkTrue(!RecognizeTerm(r, "(1|1, sd = 0.5)"))
  checkTrue(RecognizeTerm(r, "(1|x:y)"))
}

TestScaledRandEffect <- function() {
  s <- ScaledPoissonTerm$new()
  checkTrue(inherits(s, "ScaledPoissonTerm"))
  checkTrue(RecognizeTerm(s, "(y|x)"))
  checkTrue(RecognizeTerm(s, "(y| 1 )"))
  checkTrue(RecognizeTerm(s, "(y|1)"))
  checkTrue(RecognizeTerm(s, "(y|1, sd = 0.5)"))
  checkTrue(!RecognizeTerm(s, "(y x, sd = 0.5)"))
  checkTrue(!RecognizeTerm(s, "(1|x)"))
  checkTrue(!RecognizeTerm(s, "(1|1)"))
  checkTrue(!RecognizeTerm(s, "(1| 1)"))
  checkTrue(!RecognizeTerm(s, "(1|1, sd = 0.5)"))
}

TestGaussianRandEffect <- function() {
  g <- GaussianRanefTerm$new()
  checkTrue(inherits(g, "GaussianRanefTerm"))
  checkTrue(RecognizeTerm(g, "(1|x)"))
  checkTrue(RecognizeTerm(g, "(1|x, sd = 0.3)"))
  checkTrue(!RecognizeTerm(g, "(y|x)"))
  checkTrue(RecognizeTerm(g, "(1|x:y)"))
}

TestGaussianNoiseVariance <- function() {
  g <- GaussianNoiseVarianceTerm$new()
  checkTrue(inherits(g, "GaussianNoiseVarianceTerm"))
  checkTrue(RecognizeTerm(g, "stddev(n)"))
  checkTrue(RecognizeTerm(g, "sd(n)"))
  checkTrue(!RecognizeTerm(g, "offset(n)"))
}

