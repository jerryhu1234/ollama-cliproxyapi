package chatcompletions

import (
	"encoding/json"
	"strings"
	"testing"
)

// Vendor thinking spellings, all model-dependent upstream: a bare reasoning
// string, reasoning_details[].text (which mirrors the same token), and the
// standard reasoning_content. The kernel must read all three without
// double-counting a mirrored value.
const (
	thinkRole    = `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}`
	thinkDetails = `data: {"choices":[{"index":0,"delta":{"reasoning":"We","reasoning_details":[{"type":"reasoning.text","text":"We","format":"unknown","index":0}]}}]}`
	thinkPlain   = `data: {"choices":[{"index":0,"delta":{"reasoning":" need"}}]}`
	thinkStd     = `data: {"choices":[{"index":0,"delta":{"reasoning_content":" only"}}]}`
	thinkText    = `data: {"choices":[{"index":0,"delta":{"content":"2"}}]}`
	thinkFinish  = `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
)

func thinkingDeltas(evs []sseEvt) string {
	var sb strings.Builder
	for _, e := range evs {
		if e.Name != "content_block_delta" {
			continue
		}
		d := e.Data["delta"].(map[string]any)
		if d["type"] == "thinking_delta" {
			sb.WriteString(d["thinking"].(string))
		}
	}
	return sb.String()
}

// TestStreamReasoningToClaude pins the claude leg: vendor thinking text opens
// the leading thinking block at index 0 (no signature — the upstream carries
// none), the text block follows at index 1, and the mirrored reasoning field
// is counted once.
func TestStreamReasoningToClaude(t *testing.T) {
	sc := NewStreamConverter("claude")
	evs := feedAll(t, sc, thinkRole, thinkDetails, thinkPlain, thinkStd, thinkText, thinkFinish)
	evs = append(evs, feedAll(t, sc, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`)...)
	raw, done, eErr := sc.Feed([]byte("data: [DONE]\n\n"))
	if eErr != nil || !done {
		t.Fatalf("[DONE] must finish the stream: %v", eErr)
	}
	evs = append(evs, parseEvents(t, raw)...)

	names := make([]string, len(evs))
	for i, e := range evs {
		names[i] = e.Name
	}
	want := []string{
		"message_start",
		"content_block_start", // thinking @0
		"content_block_delta",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",  // thinking closed before text opens
		"content_block_start", // text @1
		"content_block_delta",
		"content_block_stop", // finish closes the text block
		"message_delta",
		"message_stop",
	}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("event sequence = %v, want %v", names, want)
	}

	start := evs[1].Data["content_block"].(map[string]any)
	if evs[1].Data["index"] != float64(0) || start["type"] != "thinking" || start["thinking"] != "" {
		t.Fatalf("thinking block start wrong: %v", evs[1])
	}
	if _, ok := start["signature"]; ok {
		t.Fatalf("thinking block must carry no signature: %v", start)
	}
	if evs[5].Data["index"] != float64(0) || evs[6].Data["index"] != float64(1) ||
		evs[6].Data["content_block"].(map[string]any)["type"] != "text" {
		t.Fatalf("thinking must close before the text block opens: %v", evs[5:7])
	}
	if got := thinkingDeltas(evs); got != "We need only" {
		t.Fatalf("thinking text = %q, want %q", got, "We need only")
	}
}

// TestStreamReasoningToClaudeThinkingOnly pins a reasoning-only stream: the
// thinking block still opens and closes (no phantom text block).
func TestStreamReasoningToClaudeThinkingOnly(t *testing.T) {
	sc := NewStreamConverter("claude")
	evs := feedAll(t, sc, thinkRole, thinkDetails, thinkFinish)
	if got := thinkingDeltas(evs); got != "We" {
		t.Fatalf("thinking text = %q", got)
	}
	for _, e := range evs {
		if e.Name != "content_block_start" {
			continue
		}
		if e.Data["content_block"].(map[string]any)["type"] != "thinking" {
			t.Fatalf("only a thinking block may open: %v", e)
		}
	}
}

// TestStreamReasoningToResponses pins the responses leg: the reasoning item is
// announced at output index 0 with its summary part, deltas stream, the item
// closes before the message item is announced at index 1, and the terminal
// output keeps reasoning first (the same order the indexes promised).
func TestStreamReasoningToResponses(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	evs := feedAll(t, sc, thinkRole, thinkDetails, thinkPlain, thinkText, thinkFinish)
	evs = append(evs, feedAll(t, sc, `data: {"choices":[{"delta":{},"finish_reason":"length"}]}`)...)

	names := make([]string, len(evs))
	for i, e := range evs {
		names[i] = e.Name
	}
	want := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added", // reasoning @0
		"response.reasoning_summary_part.added",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.done", // closed by the text delta
		"response.reasoning_summary_part.done",
		"response.output_item.done",
		"response.output_item.added", // message @1
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("event sequence = %v, want %v", names, want)
	}

	added := evs[2].Data
	item := added["item"].(map[string]any)
	if added["output_index"] != float64(0) || item["type"] != "reasoning" ||
		item["id"] != "rs_c1_0" || item["status"] != "in_progress" {
		t.Fatalf("reasoning item announcement wrong: %v", added)
	}
	if summary, ok := item["summary"].([]any); !ok || len(summary) != 0 {
		t.Fatalf("announced summary must be empty: %v", item)
	}
	part := evs[3].Data
	if part["item_id"] != "rs_c1_0" || part["output_index"] != float64(0) || part["summary_index"] != float64(0) ||
		part["part"].(map[string]any)["type"] != "summary_text" {
		t.Fatalf("summary part announcement wrong: %v", part)
	}
	if d := evs[4].Data; d["delta"] != "We" || d["item_id"] != "rs_c1_0" || d["summary_index"] != float64(0) {
		t.Fatalf("first summary delta wrong: %v", d)
	}
	if d := evs[6].Data; d["text"] != "We need" {
		t.Fatalf("summary done must carry the accumulated text: %v", d)
	}
	if done := evs[8].Data["item"].(map[string]any); done["type"] != "reasoning" ||
		done["summary"].([]any)[0].(map[string]any)["text"] != "We need" {
		t.Fatalf("reasoning item done wrong: %v", evs[7].Data)
	}
	if msg := evs[9].Data; msg["output_index"] != float64(1) {
		t.Fatalf("message must follow the reasoning item: %v", msg)
	}

	// The first finish_reason (stop) is the one held: the post-finish line's
	// own finish_reason is ignored as a repeat.
	completed := evs[len(evs)-1].Data["response"].(map[string]any)
	if completed["status"] != "completed" {
		t.Fatalf("status = %v, want completed", completed["status"])
	}
	output := completed["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("terminal output = %v, want reasoning + message", output)
	}
	rs := output[0].(map[string]any)
	if rs["type"] != "reasoning" || rs["id"] != "rs_c1_0" || rs["encrypted_content"] != "" {
		t.Fatalf("terminal reasoning item wrong: %v", rs)
	}
	if rs["summary"].([]any)[0].(map[string]any)["text"] != "We need" {
		t.Fatalf("terminal reasoning summary wrong: %v", rs)
	}
	msg := output[1].(map[string]any)
	if msg["type"] != "message" {
		t.Fatalf("terminal message wrong: %v", msg)
	}
	if part := msg["content"].([]any)[0].(map[string]any); part["text"] != "2" || part["type"] != "output_text" {
		t.Fatalf("terminal message content wrong: %v", part)
	}
}

// TestStreamReasoningPassthroughBackfill pins the openai leg: the vendor
// spellings are mirrored onto reasoning_content, vendor fields stay intact,
// and a chunk without thinking text is forwarded byte-identical.
func TestStreamReasoningPassthroughBackfill(t *testing.T) {
	sc := NewStreamConverter("openai")
	evs, done, eErr := sc.Feed([]byte(thinkDetails + "\n"))
	if eErr != nil || done || len(evs) != 1 {
		t.Fatalf("passthrough feed: evs=%d done=%v err=%v", len(evs), done, eErr)
	}
	var chunk map[string]any
	if err := json.Unmarshal(evs[0], &chunk); err != nil {
		t.Fatalf("passthrough payload not JSON: %v", err)
	}
	delta := chunk["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	if delta["reasoning_content"] != "We" || delta["reasoning"] != "We" {
		t.Fatalf("reasoning_content not backfilled: %v", delta)
	}
	if details, ok := delta["reasoning_details"].([]any); !ok || len(details) != 1 {
		t.Fatalf("vendor reasoning_details dropped: %v", delta)
	}

	// Byte-identical when nothing carries thinking text, and never overwritten
	// when the standard member is already present.
	plain := []byte(`data: {"choices":[{"index":0,"delta":{"content":"2"}}]}` + "\n")
	evs, _, eErr = sc.Feed(plain)
	if eErr != nil || string(evs[0]) != `{"choices":[{"index":0,"delta":{"content":"2"}}]}` {
		t.Fatalf("non-reasoning chunk altered: %s err=%v", evs[0], eErr)
	}
}

// TestNonStreamReasoningTargets pins the non-stream legs: the claude thinking
// block and the responses reasoning item lead the output, and the openai
// passthrough gains reasoning_content.
func TestNonStreamReasoningTargets(t *testing.T) {
	body := `{"id":"r1","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"2","reasoning":"We need only"}}]}`

	t.Run("claude", func(t *testing.T) {
		m := mustConvert(t, "claude", body)
		blocks := m["content"].([]any)
		if len(blocks) != 2 {
			t.Fatalf("blocks = %v, want thinking + text", blocks)
		}
		think := blocks[0].(map[string]any)
		if think["type"] != "thinking" || think["thinking"] != "We need only" {
			t.Fatalf("thinking block wrong: %v", think)
		}
		if _, ok := think["signature"]; ok {
			t.Fatalf("thinking block must carry no signature: %v", think)
		}
		if blocks[1].(map[string]any)["type"] != "text" {
			t.Fatalf("text block must follow: %v", blocks[1])
		}
	})

	t.Run("responses", func(t *testing.T) {
		m := mustConvert(t, "openai-response", body)
		output := m["output"].([]any)
		if len(output) != 2 {
			t.Fatalf("output = %v, want reasoning + message", output)
		}
		rs := output[0].(map[string]any)
		if rs["type"] != "reasoning" || rs["id"] != "rs_r1_0" || rs["encrypted_content"] != "" {
			t.Fatalf("reasoning item wrong: %v", rs)
		}
		summary := rs["summary"].([]any)
		if len(summary) != 1 || summary[0].(map[string]any)["type"] != "summary_text" ||
			summary[0].(map[string]any)["text"] != "We need only" {
			t.Fatalf("reasoning summary wrong: %v", summary)
		}
		if output[1].(map[string]any)["type"] != "message" {
			t.Fatalf("message item must follow: %v", output[1])
		}
	})

	t.Run("details only", func(t *testing.T) {
		m := mustConvert(t, "openai-response",
			`{"id":"r2","choices":[{"message":{"role":"assistant","content":"x",`+
				`"reasoning_details":[{"type":"reasoning.text","text":"a"},{"text":"b"}]}}]}`)
		rs := m["output"].([]any)[0].(map[string]any)
		if got := rs["summary"].([]any)[0].(map[string]any)["text"]; got != "ab" {
			t.Fatalf("details summary = %q, want %q", got, "ab")
		}
	})

	t.Run("openai passthrough", func(t *testing.T) {
		out, eErr := ConvertNonStreamResponse("openai", 200, []byte(body))
		if eErr != nil {
			t.Fatalf("unexpected error: %v", eErr)
		}
		m := decodeOut(t, out, nil)
		msg := m["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
		if msg["reasoning_content"] != "We need only" || msg["reasoning"] != "We need only" {
			t.Fatalf("message reasoning_content not backfilled: %v", msg)
		}
		if msg["content"] != "2" {
			t.Fatalf("content altered: %v", msg)
		}
	})
}
