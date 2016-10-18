index.writer.module <- Module("mod_indexer_utils",
                              PACKAGE = "emre")
ranef.updater.module <- Module("mod_ranef_updater",
                               PACKAGE = "emre")

.onLoad <- function(lib, pkg) {
  # Automatically load shared libraries
  tryCatch(library.dynam("emre", pkg, lib),
           error = function(e) {
             stop(e)
           })
  # Make sure that these protos are available
  proto.dir <- system.file("proto", package = "emre")
  RProtoBuf::readProtoFiles(dir = proto.dir)
}

EmreDebugPrint <- function(...) { }
