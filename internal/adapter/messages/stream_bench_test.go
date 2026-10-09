package messages

import (
	"encoding/json"
	"strings"
	"testing"
)

// Benchmarks pinning the SSE hot path: the per-token content_block_delta
// frame is the highest-volume payload on conversion targets, so its decode
// cost dominates Feed throughput.

var benchTextDeltaPayload = []byte(`{"index":0,"delta":{"type":"text_delta","text":"hej"}}`)

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
		var ev sseEvent
		if err := json.Unmarshal(benchTextDeltaPayload, &ev); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkStreamFeed measures a full representative conversion pass:
// message_start, 256 text deltas, tool fragment, terminal delta, stop.
func BenchmarkStreamFeed(b *testing.B) {
	var sb strings.Builder
	sb.WriteString("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"m\",\"usage\":{\"input_tokens\":10}}}\n\n")
	for i := 0; i < 256; i++ {
		sb.WriteString("event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hej\"}}\n\n")
	}
	sb.WriteString("event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_a\",\"name\":\"f\"}}\n\n")
	sb.WriteString("event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":\"}}\n\n")
	sb.WriteString("event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":9}}\n\n")
	sb.WriteString("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
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
