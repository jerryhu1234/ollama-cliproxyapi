// Package buildinfo contains metadata injected into release binaries.
package buildinfo

// Version is the plugin version reported to CLIProxyAPI. Release builds set
// this to the tag without its leading "v" using -ldflags -X.
var Version = "0.1.0-dev.1"

// Commit identifies the source revision without changing Version's semver
// shape. Release and CI builds set it independently using -ldflags -X.
var Commit = "unknown"
