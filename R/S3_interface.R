# use the S3 interface to proto-type functions that respond differently
# to its argument signature. The actual functions are implemented in
# other files.

suppressWarnings({
  GetFamilyNames <- function(x, ...) UseMethod("GetFamilyNames")
  GetRanefs <- function(x, ...) UseMethod("GetRanefs")
  GetPrior <- function(x, ...) UseMethod("GetPrior")
})  # NOLINT
