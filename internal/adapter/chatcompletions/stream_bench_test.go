package chatcompletions

import (
	"strings"
	"testing"
)

// Benchmarks pinning the DEFAULT route's chunk path. Every Ollama request
// without a route override lands on chat/completions with the openai target,
// so passthroughLine runs once per upstream SSE frame — the hottest line in
// the plugin. The two cases are the ones whose cost is easy to regress:
//
//   - content/usage frames: one typed decode, no reasoning mirror (the
//     common case; the gjson re-validation must stay off this path).
//   - reasoning frames: the same decode plus the reasoning_content mirror,
//     which needs sjson's whole-payload rewrite.

var (
	benchContentPayload = "data: {\"id\":\"c1\",\"model\":\"m\",\"created\":1,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"token text\"}}]}"
	benchUsagePayload   = "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16}}"
	benchThinkPayload   = "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"we need\",\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"we need\"}]}}]}"
)

// BenchmarkPassthroughLine is the steady-state content frame: decode once,
// observe, forward byte-identical.
func BenchmarkPassthroughLine(b *testing.B) {
	b.ReportAllocs()
	sc := NewStreamConverter("openai")
	for i := 0; i < b.N; i++ {
		evs, eErr := sc.handleLine(benchContentPayload)
		if eErr != nil || len(evs) != 1 {
			b.Fatalf("passthrough broken: events=%d err=%v", len(evs), eErr)
		}
	}
}

// BenchmarkPassthroughLineUsage is the include_usage trailer: a choice-less
// frame, so the observation branch is skipped but the decode still runs.
func BenchmarkPassthroughLineUsage(b *testing.B) {
	b.ReportAllocs()
	sc := NewStreamConverter("openai")
	for i := 0; i < b.N; i++ {
		evs, eErr := sc.handleLine(benchUsagePayload)
		if eErr != nil || len(evs) != 1 {
			b.Fatalf("passthrough broken: events=%d err=%v", len(evs), eErr)
		}
	}
}

// BenchmarkPassthroughLineBackfill is the reasoning frame where the mirror
// actually fires (sjson rewrite of the payload).
func BenchmarkPassthroughLineBackfill(b *testing.B) {
	b.ReportAllocs()
	sc := NewStreamConverter("openai")
	for i := 0; i < b.N; i++ {
		evs, eErr := sc.handleLine(benchThinkPayload)
		if eErr != nil || len(evs) != 1 {
			b.Fatalf("passthrough broken: events=%d err=%v", len(evs), eErr)
		}
		if !strings.Contains(string(evs[0]), `"reasoning_content":"we need"`) {
			b.Fatalf("backfill did not fire: %s", evs[0])
		}
	}
}

// BenchmarkFeedContentStream measures a full default-route pass: 64 content
// frames, a usage trailer, and [DONE] through the line-splitting Feed path.
func BenchmarkFeedContentStream(b *testing.B) {
	var sb strings.Builder
	sb.WriteString("data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n")
	for i := 0; i < 64; i++ {
		sb.WriteString("data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"token text\"}}]}\n")
	}
	sb.WriteString("data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":4}}\n")
	sb.WriteString("data: [DONE]\n")
	stream := []byte(sb.String())

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sc := NewStreamConverter("openai")
		events, done, eErr := sc.Feed(stream)
		if eErr != nil || !done || len(events) != 66 {
			b.Fatalf("feed broken: done=%v events=%d err=%v", done, len(events), eErr)
		}
	}
}
