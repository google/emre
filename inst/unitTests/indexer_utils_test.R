TestBiasIndex <- function() {
  kNumObservations <- 999

  bias.index <- .CreateBiasIndexReader(999)
  checkEquals(typeof(bias.index), "externalptr")
  checkTrue(inherits(bias.index, "IndexReader"))
  checkTrue(is(bias.index, "BiasIndexReader"))

  num.levels <- .IndexerNumLevels(bias.index)
  checkEquals(1, num.levels)

  num.obs <- .IndexerNumObservations(bias.index)
  checkEquals(kNumObservations, num.obs)

  row2levelid.map <- .IndexerRowToLevelMap(bias.index)
  checkEquals(kNumObservations, length(row2levelid.map))
  checkTrue(all(row2levelid.map == 0))
}

TestWriteAndReadStringFeaturesMemoryIndex <- function() {
  kFeatureFamily <- "test_feature"
  kNumFeatures <- 500
  kNumObservations <- 2 * kNumFeatures

  writer.handle <- .CreateIndexWriter(kFeatureFamily)
  levels <- paste("feature", 1:kNumFeatures, sep = "_")
  .IndexerWriteStringFeatures(writer.handle, levels)
  # we have a second set of observations with the same levels
  .IndexerWriteStringFeatures(writer.handle, levels)
  index.reader <- writer.handle$close()

  num.levels <- .IndexerNumLevels(index.reader)
  checkEquals(kNumFeatures, num.levels)

  string.levels <- .IndexerStringLevels(index.reader)
  checkEquals(kNumFeatures, length(string.levels))
  checkTrue(all(string.levels %in% levels))

  row.scaling <- .IndexerRowScaling(index.reader)
  checkEquals(kNumObservations, length(row.scaling))
  checkEquals(rep(1, kNumObservations), row.scaling)

  num.obs <- .IndexerNumObservations(index.reader)
  checkEquals(kNumObservations, num.obs)

  row2levelid.map <- .IndexerRowToLevelMap(index.reader)
  checkEquals(kNumObservations, length(row2levelid.map))
}

TestWriteAndReadStringFeatureWithScalingMemoryIndex <- function() {
  kFeatureFamily <- "test_feature"
  kNumFeatures <- 500
  kNumObservations <- 2 * kNumFeatures

  writer.handle <- .CreateIndexWriter(kFeatureFamily)
  levels <- paste("feature", 1:kNumFeatures, sep = "_")
  scaling <- seq(from = 0.001, to = 1.0, length.out = 2 * kNumFeatures)

  .IndexerWriteStringFeatures(writer.handle, levels,
                              scaling = scaling[1:500])
  # we have a second set of observations with same levels but different scaling
  .IndexerWriteStringFeatures(writer.handle, levels,
                              scaling = scaling[501:1000])
  index.reader <- writer.handle$close()

  num.levels <- .IndexerNumLevels(index.reader)
  checkEquals(kNumFeatures, num.levels)

  string.levels <- .IndexerStringLevels(index.reader)
  checkEquals(kNumFeatures, length(string.levels))
  checkTrue(all(string.levels %in% levels))

  row.scaling <- .IndexerRowScaling(index.reader)
  checkEquals(kNumObservations, length(row.scaling))
  checkEquals(scaling, row.scaling)

  num.obs <- .IndexerNumObservations(index.reader)
  checkEquals(kNumObservations, num.obs)

  row2levelid.map <- .IndexerRowToLevelMap(index.reader)
  checkEquals(kNumObservations, length(row2levelid.map))
}
