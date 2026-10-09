package responses

import (
	"encoding/json"
	"strings"
	"testing"
)

// Benchmarks pinning the SSE hot path: response.output_text.delta is the
// highest-volume frame on conversion targets, so its decode cost dominates
// Feed throughput.

var benchTextDeltaPayload = []byte(`{"delta":"hej"}`)

func BenchmarkDeltaDecodeGeneric(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var m map[string]any
		if err := json.Unmarshal(benchTextDeltaPayload, &m); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDeltaDecodeTyped(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var ev textDeltaEvent
		if err := json.Unmarshal(benchTextDeltaPayload, &ev); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkStreamFeed measures a full representative conversion pass:
// created, 256 text deltas, an argument fragment, completion.
func BenchmarkStreamFeed(b *testing.B) {
	var sb strings.Builder
	sb.WriteString("event: response.created\ndata: {\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5.6-luna\",\"created_at\":1700000000}}\n\n")
	for i := 0; i < 256; i++ {
		sb.WriteString("event: response.output_text.delta\ndata: {\"delta\":\"hej\"}\n\n")
	}
	sb.WriteString("event: response.function_call_arguments.delta\ndata: {\"item_id\":\"fc_1\",\"delta\":\"{\\\"a\\\":\"}\n\n")
	sb.WriteString("event: response.completed\ndata: {\"response\":{\"usage\":{\"input_tokens\":4,\"output_tokens\":9}}}\n\n")
	stream := []byte(sb.String())

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sc := NewStreamConverter("openai")
		events, done, eErr := sc.Feed(stream)
		if eErr != nil || !done || len(events) == 0 {
			b.Fatalf("feed broken: done=%v events=%d err=%v", done, len(events), eErr)
		}
	}
}
