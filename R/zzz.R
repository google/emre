index.writer.module <- Module("mod_indexer_utils",
                              PACKAGE = "_indexer_utils")
ranef.updater.module <- Module("mod_ranef_updater",
                               PACKAGE = "_ranef_updater")

.onLoad <- function(lib, pkg) {
  # Automatically load shared libraries
  nativesupport::LoadNativeExtension("_indexer_utils", pkg, lib,
                      "contentads/analysis/caa/search_plus/regmh/emre")
  nativesupport::LoadNativeExtension("_ranef_updater", pkg, lib,
                      "contentads/analysis/caa/search_plus/regmh/emre")
  # Make sure that these protos are available
  rglib::ReadProtoFilesFromResources(
      "contentads/analysis/caa/search_plus/regmh/emre/src/training_data.proto")
}

EmreDebugPrint <- function(...) { }
