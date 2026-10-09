//go:build !debug

package plugin

// debugTrace is compiled as an empty no-op stub for release/production builds.
func debugTrace(string, ...any) {}
