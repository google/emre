.CreateIndexWriter <- function(feature.family) {
  # Creates a new in memory index writer S4 object
  #
  # Args:
  #   feature.family: A string with the feature family name
  #
  # Returns:
  #   An index writer S4 object
  stopifnot(is.character(feature.family))

  index.writer <- index.writer.module$FeatureIndexMemoryWriter
  writer.handle <- new(index.writer, feature.family)

  return(writer.handle)
}

.CreateBiasIndexReader <- function(num.observations) {
  stopifnot(is.numeric(num.observations),
            num.observations > 0)
  return(.Call("create_bias_index_reader", num.observations,
               PACKAGE = "_indexer_utils"))
}

.IndexerWriteStringFeatures <- function(writer.handle, feature.levels,
                                        scaling = NULL) {
  # Add string features (with scaling) to the index
  #
  # Args:
  #   writer.handle: An index writer S4 object
  #   feature.levels: vector with string levels
  #   scaling: (optional) vector with numeric values for scaling
  # Returns:
  #   Nothing
  stopifnot(class(writer.handle) == "Rcpp_FeatureIndexMemoryWriter",
            is.character(feature.levels) || is.factor(feature.levels),
            is.null(scaling) || is.numeric(scaling))

  # we should have a valid writer object here
  if (is.null(scaling)) {
    writer.handle$write.string.features(feature.levels)
  } else {
    writer.handle$write.scaled.string.features(feature.levels,
                                               as.double(scaling))
  }
}

.IndexerStringLevels <- function(index.reader) {
  # Returns a vector of all string feature levels
  #
  # Args:
  #   index.reader: An S4 object with the index reader
  # Returns:
  #   A character vector with the feature level names
  stopifnot(typeof(index.reader) == "externalptr",
            inherits(index.reader, "IndexReader"))
  return(.Call("get_string_levels", index.reader, PACKAGE = "_indexer_utils"))
}

.IndexerRowToLevelMap <- function(index.reader) {
  # Returns a vector the size of observations with the feature id
  #
  # Args:
  #   index.reader: An S4 object with the index reader
  # Returns:
  #   A integer vector with the feature level ids per observation (row)
  stopifnot(typeof(index.reader) == "externalptr",
            inherits(index.reader, "IndexReader"))
  return(.Call("get_levelid_map", index.reader, PACKAGE = "_indexer_utils"))
}

.IndexerRowScaling <- function(index.reader) {
  # Returns a vector the size of observations with numerical scalings
  #
  # Args:
  #   index.reader: An S4 object with the index reader
  # Returns:
  #   A numeric vector with the scaling per observation (row)
  stopifnot(typeof(index.reader) == "externalptr",
            inherits(index.reader, "IndexReader"))
  return(.Call("get_row_scaling", index.reader, PACKAGE = "_indexer_utils"))
}

.IndexerNumLevels <- function(index.reader) {
  # Returns the number of feature levels.
  #
  # Args:
  #   index.reader: An S4 object with the index reader
  # Returns:
  #   The number of feature levels
  stopifnot(typeof(index.reader) == "externalptr",
            inherits(index.reader, "IndexReader"))
  return(.Call("get_num_levels", index.reader, PACKAGE = "_indexer_utils"))
}

.IndexerNumObservations <- function(index.reader) {
  # Returns the number of observations for this indexer.
  #
  # Args:
  #   index.reader: An S4 object with the index reader
  # Returns:
  #   The number of rows of data
  stopifnot(typeof(index.reader) == "externalptr",
            inherits(index.reader, "IndexReader"))
  return(.Call("get_num_observations", index.reader,
               PACKAGE = "_indexer_utils"))
}
