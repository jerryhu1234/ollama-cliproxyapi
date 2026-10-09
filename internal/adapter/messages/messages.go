// Package messages implements the Anthropic Messages protocol adapter
// (arch §8): upstream endpoint /v1/messages serving the Claude-family cloud
// routes (FR-004). Request translation is FR-005, response and stream
// translation FR-006, acceptance coverage AC §C.
//
// Per arch §8 this package is isolated from the scheduler and catalog
// manager and uses only stdlib JSON primitives.
package messages

import (
	"net/http"

	"ollama-cliproxyapi/internal/catalog"
)

// EndpointPath is the upstream endpoint path for the Messages route (FR-004).
var EndpointPath = catalog.RouteMessages.EndpointPath()

// anthropicVersion is the API version header value the upstream expects.
const anthropicVersion = "2023-06-01"

// AuthHeaders returns the upstream authentication headers for a Messages
// request. Ollama's Anthropic-compatible endpoint accepts Authorization:
// Bearer only (x-api-key is rejected), plus the required anthropic-version
// header. The executor wires these in M5; the key is never logged
// (FR-007, FR-011).
func AuthHeaders(key string) http.Header {
	h := make(http.Header)
	h.Set("Authorization", "Bearer "+key)
	h.Set("anthropic-version", anthropicVersion)
	return h
}
