# All functions matching the regular expression "^Test" are interpreted and run
# as unit tests by default. You can change this with a parameter to
# RunUnitTests().

TestFixEfTerm <- function() {
  f <- emre:::FixefTerm$new()
  checkTrue(inherits(f, "FixefTerm"))
  checkTrue(emre:::RecognizeTerm(f, "x"))
  checkTrue(!emre:::RecognizeTerm(f, "(x"))
  checkTrue(!emre:::RecognizeTerm(f, "offset(n)"))
  checkTrue(!emre:::RecognizeTerm(f, "offset( log(n))"))
}

TestOffsetTerm <- function() {
  o <- emre:::OffsetTerm$new()
  checkTrue(inherits(o, "OffsetTerm"))
  checkTrue(emre:::RecognizeTerm(o, "offset(n)"))
  checkTrue(emre:::RecognizeTerm(o, "offset( log(n))"))
  checkTrue(emre:::RecognizeTerm(o, "offset( log(n) )"))
  checkTrue(emre:::RecognizeTerm(o, "offset(1)"))
  # unfortunately, regex doesn't deal with unbalanced paranthesis
  checkTrue(emre:::RecognizeTerm(o, "offset( log(n)"))
  # checkFalse would be more appropriate
  checkTrue(!emre:::RecognizeTerm(o, "offset"))
  checkTrue(!emre:::RecognizeTerm(o, "offset (n)"))
  checkTrue(!emre:::RecognizeTerm(o, "offset( n"))
}

TestLogOffsetTerm <- function() {
  o <- emre:::LogOffsetTerm$new()
  checkTrue(inherits(o, "OffsetTerm"))
  checkTrue(emre:::RecognizeTerm(o, "log.offset(n)"))
  checkTrue(emre:::RecognizeTerm(o, "log.offset( log(n))"))
  checkTrue(emre:::RecognizeTerm(o, "log.offset( log(n) )"))
  checkTrue(emre:::RecognizeTerm(o, "log.offset(1)"))
  # unfortunately, regex doesn't deal with unbalanced paranthesis
  checkTrue(emre:::RecognizeTerm(o, "log.offset( log(n)"))
  # checkFalse would be more appropriate
  checkTrue(!emre:::RecognizeTerm(o, "offset(1)"))
  checkTrue(!emre:::RecognizeTerm(o, "offset (n)"))
  checkTrue(!emre:::RecognizeTerm(o, "offset( n"))
}

TestInterceptTerm <- function() {
  i <- emre:::InterceptTerm$new()
  checkTrue(inherits(i, "InterceptTerm"))
  checkTrue(emre:::RecognizeTerm(i, "0"))
  checkTrue(emre:::RecognizeTerm(i, "1"))
  # checkFalse would be more appropriate
  checkTrue(!emre:::RecognizeTerm(i, "2"))
  checkTrue(!emre:::RecognizeTerm(i, "(1|x)"))
  checkTrue(!emre:::RecognizeTerm(i, "(1)"))
}

TestRandEffect <- function() {
  r <- emre:::RanefTerm$new()
  checkTrue(inherits(r, "RanefTerm"))
  checkTrue(emre:::RecognizeTerm(r, "(1|x)"))
  checkTrue(emre:::RecognizeTerm(r, "(1|x, sd = 0.5)"))
  checkTrue(!emre:::RecognizeTerm(r, "(y|x)"))
  checkTrue(!emre:::RecognizeTerm(r, "(1|1)"))
  checkTrue(!emre:::RecognizeTerm(r, "(1| 1)"))
  checkTrue(!emre:::RecognizeTerm(r, "(1|1, sd = 0.5)"))
  checkTrue(emre:::RecognizeTerm(r, "(1|x:y)"))
}

TestScaledRandEffect <- function() {
  s <- emre:::ScaledPoissonTerm$new()
  checkTrue(inherits(s, "ScaledPoissonTerm"))
  checkTrue(emre:::RecognizeTerm(s, "(y|x)"))
  checkTrue(emre:::RecognizeTerm(s, "(y| 1 )"))
  checkTrue(emre:::RecognizeTerm(s, "(y|1)"))
  checkTrue(emre:::RecognizeTerm(s, "(y|1, sd = 0.5)"))
  checkTrue(!emre:::RecognizeTerm(s, "(y x, sd = 0.5)"))
  checkTrue(!emre:::RecognizeTerm(s, "(1|x)"))
  checkTrue(!emre:::RecognizeTerm(s, "(1|1)"))
  checkTrue(!emre:::RecognizeTerm(s, "(1| 1)"))
  checkTrue(!emre:::RecognizeTerm(s, "(1|1, sd = 0.5)"))
}

TestGaussianRandEffect <- function() {
  g <- emre:::GaussianRanefTerm$new()
  checkTrue(inherits(g, "GaussianRanefTerm"))
  checkTrue(emre:::RecognizeTerm(g, "(1|x)"))
  checkTrue(emre:::RecognizeTerm(g, "(1|x, sd = 0.3)"))
  checkTrue(!emre:::RecognizeTerm(g, "(y|x)"))
  checkTrue(emre:::RecognizeTerm(g, "(1|x:y)"))
}

TestGaussianNoiseVariance <- function() {
  g <- emre:::GaussianNoiseVarianceTerm$new()
  checkTrue(inherits(g, "GaussianNoiseVarianceTerm"))
  checkTrue(emre:::RecognizeTerm(g, "stddev(n)"))
  checkTrue(emre:::RecognizeTerm(g, "sd(n)"))
  checkTrue(!emre:::RecognizeTerm(g, "offset(n)"))
}

