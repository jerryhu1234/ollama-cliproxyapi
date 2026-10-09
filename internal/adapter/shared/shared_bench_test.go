package shared

import (
	"encoding/json"
	"testing"
)

// benchPayload is a representative chat-completions-sized stream event:
// one text delta chunk with id/model/created envelope fields.
var benchPayload = map[string]any{
	"id":      "chatcmpl-20260825120000",
	"object":  "chat.completion.chunk",
	"created": 1770000000,
	"model":   "commandcode/gpt-5.6-luna",
	"choices": []any{map[string]any{
		"index":         0,
		"delta":         map[string]any{"content": "The quick brown fox jumps over the lazy dog."},
		"finish_reason": nil,
	}},
}

// Representative request body whose inbound model field already equals the
// bare upstream ID — the common case once clients adopt the public prefix
// or send bare IDs (the rewrite is then a pure no-op).
var benchBody = []byte(`{"model":"gpt-5.6-luna","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"Write me a haiku about latency budgets in distributed systems."}]}],"max_tokens":4096}`)

func BenchmarkSSEEvent(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = SSEEvent("content_block_delta", benchPayload)
	}
}

func BenchmarkRewriteModelIDNoop(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		out, eErr := RewriteModelID("gpt-5.6-luna", benchBody, "openai")
		if eErr != nil {
			b.Fatalf("rewrite failed: %v", eErr)
		}
		if len(out) == 0 {
			b.Fatal("empty output")
		}
	}
}

// Guard so the noop benchmark cannot silently drift into a body where the
// model field differs (which would measure the rewrite path instead).
func TestBenchBodyIsNoop(t *testing.T) {
	var probe struct {
		Model json.RawMessage `json:"model"`
	}
	if err := json.Unmarshal(benchBody, &probe); err != nil {
		t.Fatalf("bench body not json: %v", err)
	}
	want := `"gpt-5.6-luna"`
	if string(probe.Model) != want {
		t.Fatalf("bench model = %s, want %s", probe.Model, want)
	}
}

// The no-op fast path must return the ORIGINAL body slice unchanged.
func TestRewriteModelIDNoopReturnsOriginalBody(t *testing.T) {
	body := []byte(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}]}`)
	out, eErr := RewriteModelID("glm-5.2", body, "openai")
	if eErr != nil {
		t.Fatalf("noop rewrite failed: %v", eErr)
	}
	if &out[0] != &body[0] {
		t.Fatal("no-op case must return the original body, not a copy")
	}
	// A differing model still rewrites through the full map path.
	out, eErr = RewriteModelID("upstream-id", body, "openai")
	if eErr != nil || out == nil {
		t.Fatalf("rewrite: %v", eErr)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil || got["model"] != "upstream-id" {
		t.Fatalf("rewrite wrong: %v %s", err, out)
	}
	// An escape-free upstream id whose inbound bytes differ only in
	// escaping cannot exist (RawMessage stores decoded-value bytes), but
	// a quote-bearing id never matches the fast path and keeps the
	// documented unrepresentable-id error.
	if _, eErr := RewriteModelID(`a"b`, []byte(`{"model":"a\"b"}`), "openai"); eErr == nil {
		t.Fatal("quote-bearing id must keep the fall-through error")
	}
}

// SSE frame assembly must stay byte-identical while allocating fewer
// copies; these pins guard against drift in the append-chain rewrite.
func TestSSEFrameBytesUnchanged(t *testing.T) {
	payload := map[string]any{"a": 1}
	if got := string(SSEEvent("n", payload)); got != "event: n\ndata: {\"a\":1}\n\n" {
		t.Fatalf("SSEEvent = %q", got)
	}
}

func TestSSEDoneHelpers(t *testing.T) {
	for _, tc := range []struct{ in, name string }{
		{"[DONE]", "framer data"},
		{"  [DONE]  ", "padded"},
	} {
		if !IsSSEDone(tc.in) {
			t.Fatalf("%s (%q): IsSSEDone = false", tc.name, tc.in)
		}
	}
	for _, in := range []string{"", "[done]", "[DONE ]", "{\"x\":1}", "[DONEX]"} {
		if IsSSEDone(in) {
			t.Fatalf("IsSSEDone(%q) = true, want false", in)
		}
	}
}
