package chatcompletions

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ollama-cliproxyapi/internal/errclass"
)

type sseEvt struct {
	Name string
	Data map[string]any
}

// parseEvents decodes synthesized "event: X\ndata: {...}\n\n" frames.
func parseEvents(t *testing.T, events [][]byte) []sseEvt {
	t.Helper()
	var out []sseEvt
	for _, raw := range events {
		head, data, ok := strings.Cut(string(raw), "\ndata: ")
		if !ok || !strings.HasPrefix(head, "event: ") {
			t.Fatalf("malformed synthesized event: %q", raw)
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(strings.TrimSuffix(data, "\n\n")), &m); err != nil {
			t.Fatalf("event payload not JSON: %v (%q)", err, data)
		}
		out = append(out, sseEvt{Name: strings.TrimPrefix(head, "event: "), Data: m})
	}
	return out
}

func feedAll(t *testing.T, sc *StreamConverter, lines ...string) []sseEvt {
	t.Helper()
	var all [][]byte
	for _, line := range lines {
		evs, _, eErr := sc.Feed([]byte(line + "\n"))
		if eErr != nil {
			t.Fatalf("feed %q: %v", line, eErr)
		}
		all = append(all, evs...)
	}
	return parseEvents(t, all)
}

func TestStreamConverterOpenAIPassthrough(t *testing.T) {
	sc := NewStreamConverter("openai")
	frame := "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
	evs, done, eErr := sc.Feed([]byte(frame[:20]))
	if eErr != nil || done || len(evs) != 0 {
		t.Fatalf("partial line must buffer: evs=%v done=%v err=%v", evs, done, eErr)
	}
	evs, done, eErr = sc.Feed([]byte(frame[20:]))
	if eErr != nil || done {
		t.Fatalf("unexpected: done=%v err=%v", done, eErr)
	}
	if len(evs) != 1 || string(evs[0]) != `{"id":"1","choices":[{"delta":{"content":"hi"}}]}` {
		t.Fatalf("bare JSON payload not extracted: %q", evs)
	}
	evs, _, eErr = sc.Feed([]byte(": keep-alive\n"))
	if eErr != nil || len(evs) != 0 {
		t.Fatalf("keep-alive line should be skipped: %v %v", evs, eErr)
	}
	if _, done, _ = sc.Feed([]byte("data: [DONE]\n")); !done {
		t.Fatal("[DONE] must set done")
	}
}

func TestStreamConverterUnsupportedFormat(t *testing.T) {
	sc := NewStreamConverter("grpc")
	_, _, eErr := sc.Feed([]byte("data: {}\n"))
	if eErr == nil || eErr.Class != errclass.ClassUnsupported {
		t.Fatalf("want ClassUnsupported, got %+v", eErr)
	}
}

func TestStreamConverterClaude(t *testing.T) {
	sc := NewStreamConverter("claude")
	var evs []sseEvt

	evs = feedAll(t, sc,
		`data: {"id":"r1","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}],"usage":{"prompt_tokens":7}}`,
		`data: {"choices":[{"delta":{"content":"He"}}]}`)
	if len(evs) != 3 {
		t.Fatalf("want message_start + text block start + text_delta, got %v", evs)
	}
	msg := evs[0].Data["message"].(map[string]any)
	if msg["id"] != "r1" || msg["role"] != "assistant" || msg["model"] != "m" {
		t.Fatalf("message_start wrong: %v", msg)
	}
	if u := msg["usage"].(map[string]any); u["input_tokens"] != float64(7) || u["output_tokens"] != float64(0) {
		t.Fatalf("message_start usage wrong: %v", u)
	}
	if evs[1].Name != "content_block_start" || evs[1].Data["index"] != float64(0) ||
		evs[1].Data["content_block"].(map[string]any)["type"] != "text" {
		t.Fatalf("text block start wrong: %v", evs[1])
	}
	if d := evs[2].Data["delta"].(map[string]any); d["type"] != "text_delta" || d["text"] != "He" {
		t.Fatalf("text_delta wrong: %v", evs[2])
	}

	// Tool call opens: closes the text block, starts tool_use at index 1.
	evs = feedAll(t, sc, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":""}}]}}]}`)
	if len(evs) != 2 || evs[0].Name != "content_block_stop" || evs[0].Data["index"] != float64(0) {
		t.Fatalf("text block not closed before tool block: %v", evs)
	}
	blk := evs[1].Data["content_block"].(map[string]any)
	if evs[1].Data["index"] != float64(1) || blk["type"] != "tool_use" || blk["id"] != "c1" || blk["name"] != "f" {
		t.Fatalf("tool_use start wrong: %v", evs[1])
	}

	// Argument fragments accumulate per index across chunks.
	frag := feedAll(t, sc,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"k\":"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`)
	if len(frag) != 2 {
		t.Fatalf("want two input_json_delta events, got %v", frag)
	}
	if d := frag[0].Data["delta"].(map[string]any); d["type"] != "input_json_delta" || d["partial_json"] != `{"k":` {
		t.Fatalf("first fragment wrong: %v", frag[0])
	}
	if d := frag[1].Data["delta"].(map[string]any); d["partial_json"] != "1}" {
		t.Fatalf("second fragment wrong: %v", frag[1])
	}

	// Second tool index in one chunk (text already closed): the still-open
	// first tool block stops first (close-before-next-start), then start + delta.
	evs = feedAll(t, sc, `data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"g","arguments":"{\"j\":2}"}}]}}]}`)
	if len(evs) != 3 || evs[0].Name != "content_block_stop" || evs[0].Data["index"] != float64(1) ||
		evs[1].Data["index"] != float64(2) {
		t.Fatalf("second tool block wrong: %v", evs)
	}
	if blk := evs[1].Data["content_block"].(map[string]any); blk["id"] != "c2" || blk["name"] != "g" {
		t.Fatalf("second tool_use wrong: %v", evs[1])
	}

	// Empty-choices chunk is ignored.
	if evs = feedAll(t, sc, `data: {"id":"x","choices":[]}`); len(evs) != 0 {
		t.Fatalf("empty choices must emit nothing: %v", evs)
	}

	// finish_reason closes blocks; message_delta is held until the next
	// data line (the standard include_usage trailer follows finish).
	// Block 1 was already closed when block 2 started, so one stop remains.
	evs = feedAll(t, sc, `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":5}}`)
	if len(evs) != 1 {
		t.Fatalf("want one stop with message_delta deferred, got %v", evs)
	}
	if evs[0].Name != "content_block_stop" || evs[0].Data["index"] != float64(2) {
		t.Fatalf("tool block stop wrong: %v", evs)
	}

	// The post-finish line flushes the held message_delta (the line's own
	// finish_reason is ignored as a repeat).
	evs = feedAll(t, sc, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`)
	if len(evs) != 1 {
		t.Fatalf("want only the flushed message_delta, got %v", evs)
	}
	md := evs[0]
	if md.Name != "message_delta" || md.Data["delta"].(map[string]any)["stop_reason"] != "tool_use" {
		t.Fatalf("message_delta wrong: %v", md)
	}
	if u := md.Data["usage"].(map[string]any); u["input_tokens"] != float64(7) || u["output_tokens"] != float64(5) {
		t.Fatalf("message_delta usage wrong: %v", u)
	}

	// [DONE] emits message_stop and terminates.
	all, done, eErr := sc.Feed([]byte("data: [DONE]\n\n"))
	if eErr != nil || !done {
		t.Fatalf("[DONE] must finish the stream: %v", eErr)
	}
	evs = parseEvents(t, all)
	if len(evs) != 1 || evs[0].Name != "message_stop" {
		t.Fatalf("want only message_stop on [DONE], got %v", evs)
	}
}

// Pure tool-call streams (function_call deltas + done, zero content) must
// mirror non-stream chatToClaude's shape: only tool_use blocks, starting
// at index 0 — no phantom empty text block opened ahead of them (FR-006
// sibling parity).
func TestStreamConverterClaudeToolOnlyStream(t *testing.T) {
	sc := NewStreamConverter("claude")
	evs := feedAll(t, sc,
		`data: {"id":"r","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":""}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`,
		`data: [DONE]`)
	var toolStartIdx = float64(-1)
	stops := map[float64]bool{}
	for i, e := range evs {
		switch e.Name {
		case "content_block_start":
			blk := e.Data["content_block"].(map[string]any)
			if blk["type"] == "text" {
				t.Fatalf("event %d: phantom text block in tool-only stream: %v", i, e)
			}
			if blk["type"] == "tool_use" && toolStartIdx < 0 {
				toolStartIdx = e.Data["index"].(float64)
				if blk["id"] != "c1" || blk["name"] != "f" {
					t.Fatalf("tool_use start wrong: %v", e)
				}
			}
		case "content_block_stop":
			stops[e.Data["index"].(float64)] = true
		case "content_block_delta":
			if d := e.Data["delta"].(map[string]any); d["type"] != "input_json_delta" || d["partial_json"] != "{}" {
				t.Fatalf("unexpected delta in tool-only stream: %v", e)
			}
		}
	}
	if toolStartIdx != 0 {
		t.Fatalf("tool_use must start at index 0 like non-stream chatToClaude, got %v", toolStartIdx)
	}
	if !stops[0] {
		t.Fatalf("tool_use block at index 0 never stopped: %v", evs)
	}
	var md sseEvt
	for _, e := range evs {
		if e.Name == "message_delta" {
			md = e
		}
	}
	if md.Data == nil || md.Data["delta"].(map[string]any)["stop_reason"] != "tool_use" {
		t.Fatalf("stop_reason wrong: %v", md)
	}
	u := md.Data["usage"].(map[string]any)
	if u["input_tokens"] != float64(3) || u["output_tokens"] != float64(2) {
		t.Fatalf("usage wrong: %v", md)
	}
}

func TestStreamConverterClaudeInterleavedTextTool(t *testing.T) {
	sc := NewStreamConverter("claude")
	evs := feedAll(t, sc,
		`data: {"id":"r","model":"m","choices":[{"delta":{"role":"assistant"}}]}`,
		`data: {"choices":[{"delta":{"content":"a"}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{}"}}]}}]}`,
		`data: {"choices":[{"delta":{"content":"b"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`)

	// Strict per-index lifecycle: start -> delta(s) -> stop, no reuse.
	state := map[float64]int{} // 0 unseen, 1 started, 2 delta seen, 3 stopped
	types := map[float64]string{}
	deltas := map[float64][]string{}
	for _, e := range evs {
		idx, ok := e.Data["index"].(float64)
		if !ok {
			continue
		}
		switch e.Name {
		case "content_block_start":
			if state[idx] != 0 {
				t.Fatalf("content_block_start at index %v out of order: %v", idx, evs)
			}
			state[idx] = 1
			types[idx] = e.Data["content_block"].(map[string]any)["type"].(string)
		case "content_block_delta":
			if state[idx] != 1 {
				t.Fatalf("content_block_delta at index %v out of order: %v", idx, e)
			}
			state[idx] = 2
			deltas[idx] = append(deltas[idx], e.Data["delta"].(map[string]any)["type"].(string))
		case "content_block_stop":
			if state[idx] != 1 && state[idx] != 2 {
				t.Fatalf("content_block_stop at index %v out of order: %v", idx, e)
			}
			state[idx] = 3
		}
	}
	wantTypes := map[float64]string{0: "text", 1: "tool_use", 2: "text"}
	for idx, want := range wantTypes {
		if types[idx] != want || state[idx] != 3 {
			t.Fatalf("block %v: type=%q state=%d, want %q fully closed", idx, types[idx], state[idx], want)
		}
	}
	if len(state) != 3 {
		t.Fatalf("unexpected extra blocks: %v", types)
	}
	if len(deltas[0]) != 1 || deltas[0][0] != "text_delta" ||
		len(deltas[1]) != 1 || deltas[1][0] != "input_json_delta" ||
		len(deltas[2]) != 1 || deltas[2][0] != "text_delta" {
		t.Fatalf("delta types wrong: %v", deltas)
	}
}

// assertCloseBeforeStart enforces Anthropic's close-before-next-start
// discipline: at most one content block open at any time, every start
// finding zero open blocks, and the stream ending with none open.
func assertCloseBeforeStart(t *testing.T, evs []sseEvt) {
	t.Helper()
	open := map[float64]bool{}
	for i, e := range evs {
		idx, ok := e.Data["index"].(float64)
		if !ok {
			continue
		}
		switch e.Name {
		case "content_block_start":
			if len(open) != 0 {
				t.Fatalf("event %d: content_block_start for index %v while %d block(s) still open: %v", i, idx, len(open), evs[:i+1])
			}
			open[idx] = true
		case "content_block_stop":
			if !open[idx] {
				t.Fatalf("event %d: content_block_stop for unopened index %v", i, idx)
			}
			delete(open, idx)
		}
	}
	if len(open) != 0 {
		t.Fatalf("blocks left open: %v", open)
	}
}

// PoC (a): interleaved text→tool→text must never hold two blocks open —
// the reopened text block may only start after the tool block stopped.
func TestStreamConverterClaudeInterleaveClosesBlocks(t *testing.T) {
	sc := NewStreamConverter("claude")
	evs := feedAll(t, sc,
		`data: {"id":"r","model":"m","choices":[{"delta":{"content":"a"}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{}"}}]}}]}`,
		`data: {"choices":[{"delta":{"content":"b"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`)
	assertCloseBeforeStart(t, evs)
}

// PoC (b): consecutive tool_use blocks (new upstream tool_calls index while
// a sibling tool block is still streaming) must not overlap either.
func TestStreamConverterClaudeToolToolClosesBlocks(t *testing.T) {
	sc := NewStreamConverter("claude")
	evs := feedAll(t, sc,
		`data: {"id":"r","model":"m","choices":[{"delta":{"role":"assistant"}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":""}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"g","arguments":"{}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`)
	assertCloseBeforeStart(t, evs)
}

func TestStreamConverterClaudeStopReasons(t *testing.T) {
	cases := map[string]string{"stop": "end_turn", "length": "max_tokens"}
	for finish, want := range cases {
		sc := NewStreamConverter("claude")
		evs := feedAll(t, sc,
			`data: {"id":"r","model":"m","choices":[{"delta":{"role":"assistant"}}]}`,
			`data: {"choices":[{"delta":{"content":"x"}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"`+finish+`"}]}`,
			`data: [DONE]`)
		var md sseEvt
		stops := 0
		for _, e := range evs {
			if e.Name == "message_delta" {
				md = e
			}
			if e.Name == "content_block_stop" {
				stops++
			}
		}
		if md.Data == nil || md.Data["delta"].(map[string]any)["stop_reason"] != want {
			t.Fatalf("finish %q: message_delta wrong: %v", finish, evs)
		}
		// Usage is always attached (F-R6); zero-valued when upstream sent none.
		u := md.Data["usage"].(map[string]any)
		if u["input_tokens"] != float64(0) || u["output_tokens"] != float64(0) {
			t.Fatalf("finish %q: usage must default to zeros: %v", finish, md)
		}
		if stops != 1 {
			t.Fatalf("finish %q: want one text block stop, got %d", finish, stops)
		}
	}
}

func TestStreamConverterClaudeVariants(t *testing.T) {
	t.Run("choice-less usage chunk feeds the terminal message_delta", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		evs := feedAll(t, sc,
			`data: {"id":"r","model":"m","choices":[{"delta":{"role":"assistant"}}]}`,
			`data: {"choices":[{"delta":{"content":"x"}}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4}}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`data: [DONE]`)
		var md sseEvt
		for _, e := range evs {
			if e.Name == "message_delta" {
				md = e
			}
		}
		u, ok := md.Data["usage"].(map[string]any)
		if !ok || u["output_tokens"] != float64(4) {
			t.Fatalf("terminal event lost choice-less usage: %v", md)
		}
	})
	t.Run("include_usage trailer after finish feeds the terminal message_delta", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		evs := feedAll(t, sc,
			`data: {"id":"r","model":"m","choices":[{"delta":{"role":"assistant"}}]}`,
			`data: {"choices":[{"delta":{"content":"x"}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4}}`,
			`data: [DONE]`)
		mdIdx, stopIdx := -1, -1
		for i, e := range evs {
			switch e.Name {
			case "message_delta":
				mdIdx = i
			case "message_stop":
				stopIdx = i
			}
		}
		if mdIdx < 0 || stopIdx < 0 || mdIdx > stopIdx {
			t.Fatalf("message_delta must precede message_stop: %v", evs)
		}
		md := evs[mdIdx]
		if md.Data["delta"].(map[string]any)["stop_reason"] != "end_turn" {
			t.Fatalf("stop reason wrong: %v", md)
		}
		u, ok := md.Data["usage"].(map[string]any)
		if !ok || u["output_tokens"] != float64(4) {
			t.Fatalf("terminal event lost trailing include_usage usage: %v", md)
		}
	})
	t.Run("first chunk without usage starts at zero", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		evs := feedAll(t, sc, `data: {"id":"r","model":"m","choices":[{"delta":{"role":"assistant"}}]}`)
		u := evs[0].Data["message"].(map[string]any)["usage"].(map[string]any)
		if u["input_tokens"] != float64(0) {
			t.Fatalf("input_tokens not defaulted: %v", u)
		}
	})
	t.Run("abrupt DONE with no content emits no phantom text block", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		all, done, eErr := sc.Feed([]byte(
			"data: {\"id\":\"r\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\ndata: [DONE]\n\n"))
		if eErr != nil || !done {
			t.Fatalf("unexpected: %v", eErr)
		}
		evs := parseEvents(t, all)
		if len(evs) != 2 || evs[0].Name != "message_start" || evs[1].Name != "message_stop" {
			t.Fatalf("abrupt DONE sequence wrong: %v", evs)
		}
	})
	t.Run("CRLF line endings tolerated", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		evs, done, eErr := sc.Feed([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\r\n"))
		if eErr != nil || done || len(evs) != 3 {
			t.Fatalf("CRLF chunk wrong: %d events %v", len(evs), eErr)
		}
	})
	t.Run("non-data lines ignored", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		if evs, _, eErr := sc.Feed([]byte(": ping\nevent: ping\n\n")); eErr != nil || len(evs) != 0 {
			t.Fatalf("comment lines must be ignored: %v %v", evs, eErr)
		}
	})
	t.Run("malformed chunk errors with short snippet", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		payload := `{"choices":[{"delta":{"content":"` + strings.Repeat("x", 200)
		_, _, eErr := sc.Feed([]byte("data: " + payload + "\n"))
		if eErr == nil || eErr.Class != errclass.ClassTranslation {
			t.Fatalf("want ClassTranslation, got %+v", eErr)
		}
		if len(eErr.Message) > 160 || strings.Contains(eErr.Message, payload[100:]) {
			t.Fatalf("error echoes too much upstream body: %q", eErr.Message)
		}
	})
	t.Run("partial line carried across feeds", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		line := `data: {"choices":[{"delta":{"content":"hi"}}]}` + "\n"
		if evs, _, eErr := sc.Feed([]byte(line[:25])); eErr != nil || len(evs) != 0 {
			t.Fatalf("partial line must buffer: %v %v", evs, eErr)
		}
		evs, _, eErr := sc.Feed([]byte(line[25:]))
		if eErr != nil || len(evs) != 3 {
			t.Fatalf("completed line must emit: %d events %v", len(evs), eErr)
		}
	})
}

func TestStreamConverterResponses(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	var evs []sseEvt

	// The role chunk primes id/model and announces response.created (F18).
	evs = feedAll(t, sc, `data: {"id":"r1","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}`)
	if len(evs) != 2 || evs[0].Name != "response.created" || evs[1].Name != "response.in_progress" {
		t.Fatalf("first chunk must announce response.created: %v", evs)
	}
	if resp := evs[0].Data["response"].(map[string]any); resp["id"] != "r1" || resp["status"] != "in_progress" {
		t.Fatalf("created payload wrong: %v", resp)
	}
	evs = feedAll(t, sc, `data: {"choices":[{"delta":{"content":"He"}}]}`)
	if len(evs) != 3 || evs[0].Name != "response.output_item.added" || evs[1].Name != "response.content_part.added" || evs[2].Name != "response.output_text.delta" {
		t.Fatalf("text chunk must announce the message item first: %v", evs)
	}
	item := evs[0].Data["item"].(map[string]any)
	if item["type"] != "message" || item["role"] != "assistant" || evs[0].Data["output_index"] != float64(0) {
		t.Fatalf("message item wrong: %v", evs[0])
	}
	td := evs[2].Data
	if td["item_id"] != "msg_r1_0" || td["content_index"] != float64(0) || td["output_index"] != float64(0) || td["delta"] != "He" {
		t.Fatalf("output_text delta wrong: %v", evs[1])
	}

	// Tool call: registration chunk announces output_item.added (F18) after
	// the message item's output_index, then argument fragments follow.
	if evs = feedAll(t, sc, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":""}}]}}]}`); len(evs) != 4 || evs[3].Name != "response.output_item.added" {
		t.Fatalf("tool registration must announce the item first: %v", evs)
	}
	if evs[3].Data["output_index"] != float64(1) {
		t.Fatalf("tool item must follow the message item's index: %v", evs[0])
	}
	evs = feedAll(t, sc,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"j\":"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"2}"}}]}}]}`)
	if len(evs) != 2 {
		t.Fatalf("want two argument deltas, got %v", evs)
	}
	for i, want := range []string{`{"j":`, `2}`} {
		e := evs[i]
		if e.Name != "response.function_call_arguments.delta" ||
			e.Data["item_id"] != "c1" || e.Data["output_index"] != float64(1) || e.Data["delta"] != want {
			t.Fatalf("argument delta %d wrong: %v", i, e)
		}
	}

	if evs = feedAll(t, sc, `data: {"id":"x","choices":[]}`); len(evs) != 0 {
		t.Fatalf("empty choices must emit nothing: %v", evs)
	}

	// finish_reason holds response.completed; the next data line flushes
	// it (the line's own finish_reason is ignored as a repeat).
	if evs = feedAll(t, sc, `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}}`); len(evs) != 0 {
		t.Fatalf("response.completed must be deferred past finish_reason: %v", evs)
	}
	evs = feedAll(t, sc, `data: {"choices":[{"delta":{},"finish_reason":"length"}]}`)
	if len(evs) < 3 || evs[len(evs)-1].Name != "response.completed" {
		t.Fatalf("completed event wrong: %v", evs)
	}
	resp := evs[len(evs)-1].Data["response"].(map[string]any)
	if resp["id"] != "r1" || resp["object"] != "response" || resp["status"] != "completed" {
		t.Fatalf("completed response wrong: %v", resp)
	}
	u := resp["usage"].(map[string]any)
	if u["input_tokens"] != float64(7) || u["output_tokens"] != float64(5) || u["total_tokens"] != float64(12) {
		t.Fatalf("completed usage wrong: %v", u)
	}

	if _, done, eErr := sc.Feed([]byte("data: [DONE]\n")); eErr != nil || !done {
		t.Fatalf("[DONE] must finish the stream: %v", eErr)
	}
}

func TestResponsesTextLifecycleSDKShape(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	events := feedAll(t, sc,
		`data: {"id":"sdk","model":"m","choices":[{"delta":{"content":"a"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}`,
		`data: [DONE]`)
	want := []string{"response.created", "response.in_progress", "response.output_item.added", "response.content_part.added", "response.output_text.delta", "response.output_text.delta", "response.output_text.done", "response.content_part.done", "response.output_item.done", "response.completed"}
	if len(events) != len(want) {
		t.Fatalf("events=%v", events)
	}
	var last float64
	for i, event := range events {
		if event.Name != want[i] {
			t.Fatalf("event %d=%s want %s", i, event.Name, want[i])
		}
		seq := event.Data["sequence_number"].(float64)
		if seq <= last {
			t.Fatalf("sequence %v after %v", seq, last)
		}
		last = seq
	}
	created := events[0].Data["response"].(map[string]any)
	if len(created["output"].([]any)) != 0 {
		t.Fatalf("created=%v", created)
	}
	added := events[2].Data["item"].(map[string]any)
	itemID := added["id"].(string)
	if itemID == "sdk" || added["status"] != "in_progress" {
		t.Fatalf("added=%v", added)
	}
	for _, idx := range []int{3, 4, 5, 6, 7} {
		if events[idx].Data["item_id"] != itemID || events[idx].Data["content_index"] != float64(0) {
			t.Fatalf("reference mismatch: %v", events[idx])
		}
	}
	done := events[8].Data["item"].(map[string]any)
	if done["id"] != itemID || done["status"] != "completed" {
		t.Fatalf("done=%v", done)
	}
	completed := events[9].Data["response"].(map[string]any)
	output := completed["output"].([]any)[0].(map[string]any)
	if output["id"] != itemID || output["status"] != "completed" || output["content"].([]any)[0].(map[string]any)["text"] != "ab" {
		t.Fatalf("completed=%v", completed)
	}
}

func TestResponsesAlternatingSegmentsKeepDistinctItems(t *testing.T) {
	for _, tc := range []struct {
		name      string
		chunks    []string
		wantTypes []string
	}{
		{"text tool text", []string{
			`data: {"id":"alt","model":"m","choices":[{"delta":{"content":"a"}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{}"}}]}}]}`,
			`data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}`,
		}, []string{"message", "function_call", "message"}},
		{"reason text reason", []string{
			`data: {"id":"alt","model":"m","choices":[{"delta":{"reasoning":"r1"}}]}`,
			`data: {"choices":[{"delta":{"content":"t"}}]}`,
			`data: {"choices":[{"delta":{"reasoning":"r2"},"finish_reason":"stop"}]}`,
		}, []string{"reasoning", "message", "reasoning"}},
		{"text reason text", []string{
			`data: {"id":"alt","model":"m","choices":[{"delta":{"content":"a"}}]}`,
			`data: {"choices":[{"delta":{"reasoning":"r"}}]}`,
			`data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}`,
		}, []string{"message", "reasoning", "message"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sc := NewStreamConverter("openai-response")
			var all []sseEvt
			for _, chunk := range tc.chunks {
				all = append(all, feedAll(t, sc, chunk)...)
			}
			all = append(all, feedAll(t, sc, `data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`, `data: [DONE]`)...)
			var added, done []string
			var completed map[string]any
			last := float64(0)
			for _, event := range all {
				seq := event.Data["sequence_number"].(float64)
				if seq <= last {
					t.Fatalf("sequence %v", all)
				}
				last = seq
				switch event.Name {
				case "response.output_item.added":
					added = append(added, event.Data["item"].(map[string]any)["type"].(string))
				case "response.output_item.done":
					done = append(done, event.Data["item"].(map[string]any)["type"].(string))
				case "response.completed":
					completed = event.Data["response"].(map[string]any)
				}
			}
			if strings.Join(added, ",") != strings.Join(tc.wantTypes, ",") || len(done) != len(tc.wantTypes) {
				t.Fatalf("added=%v done=%v", added, done)
			}
			output := completed["output"].([]any)
			var got []string
			for _, item := range output {
				got = append(got, item.(map[string]any)["type"].(string))
			}
			if strings.Join(got, ",") != strings.Join(tc.wantTypes, ",") {
				t.Fatalf("output=%v", output)
			}
		})
	}
}

func TestResponsesToolArgumentsCanSpanOtherSegments(t *testing.T) {
	for _, middle := range []string{
		`data: {"choices":[{"delta":{"content":"middle"}}]}`,
		`data: {"choices":[{"delta":{"reasoning":"middle"}}]}`,
	} {
		sc := NewStreamConverter("openai-response")
		events := feedAll(t, sc,
			`data: {"id":"span","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{\"x\":"}}]}}]}`,
			middle,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`, `data: [DONE]`)
		var completed map[string]any
		doneCount := 0
		for _, event := range events {
			if event.Name == "response.output_item.done" && event.Data["item"].(map[string]any)["type"] == "function_call" {
				doneCount++
			}
			if event.Name == "response.completed" {
				completed = event.Data["response"].(map[string]any)
			}
		}
		if doneCount != 1 {
			t.Fatalf("done count=%d events=%v", doneCount, events)
		}
		found := false
		for _, raw := range completed["output"].([]any) {
			item := raw.(map[string]any)
			if item["type"] == "function_call" {
				found = true
				if item["arguments"] != `{"x":1}` {
					t.Fatalf("arguments=%v", item)
				}
			}
		}
		if !found {
			t.Fatal("missing function")
		}
	}
}

func TestResponsesToolCompletionGuards(t *testing.T) {
	t.Run("fragment after terminal done is ignored", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		feedAll(t, sc, `data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, `data: [DONE]`)
		events, done, e := sc.Feed([]byte(`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"secret"}}]}}]}` + "\n"))
		if e != nil || !done || len(events) != 0 {
			t.Fatalf("events=%v done=%v error=%v", events, done, e)
		}
	})
	t.Run("mixed complete incomplete fails without completed", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		events, _, e := sc.Feed([]byte(`data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"a","arguments":"{}"}},{"index":1,"id":"b","function":{"name":"b","arguments":"{\"x\":"}}]},"finish_reason":"tool_calls"}]}` + "\n"))
		if e == nil {
			t.Fatal("expected incomplete error")
		}
		if strings.Contains(string(bytes.Join(events, nil)), "response.completed") {
			t.Fatal("false completed")
		}
	})
}

func TestResponsesTerminalValidationDONEAndEOF(t *testing.T) {
	for _, end := range []string{"done", "eof"} {
		for _, tc := range []struct {
			name   string
			chunks []string
		}{
			{"incomplete function", []string{`data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{\"x\":"}}]}}]}`}},
			{"incomplete custom", []string{`data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"custom","arguments":"{\"input\":"}}]}}]}`}},
			{"unannounced prefix", []string{`data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"pre","arguments":"{}"}}]}}]}`}},
		} {
			t.Run(end+" "+tc.name, func(t *testing.T) {
				sc := NewStreamConverter("openai-response", []byte(`{"tools":[{"type":"custom","name":"custom"},{"type":"function","name":"prefix_long"}]}`))
				var emitted [][]byte
				for _, chunk := range tc.chunks {
					events, _, e := sc.Feed([]byte(chunk + "\n"))
					if e != nil {
						t.Fatal(e)
					}
					emitted = append(emitted, events...)
				}
				var e *errclass.Error
				if end == "done" {
					events, _, err := sc.Feed([]byte("data: [DONE]\n"))
					emitted = append(emitted, events...)
					e = err
				} else {
					events, err := sc.FlushWithError()
					emitted = append(emitted, events...)
					e = err
				}
				if e == nil || strings.Contains(e.Message, "{\"x\"") || strings.Contains(string(bytes.Join(emitted, nil)), "response.completed") {
					t.Fatalf("error=%v emitted=%s", e, bytes.Join(emitted, nil))
				}
			})
		}
	}
	for _, end := range []string{"done", "eof"} {
		t.Run("complete without finish "+end, func(t *testing.T) {
			sc := NewStreamConverter("openai-response")
			events, _, e := sc.Feed([]byte(`data: {"id":"x","choices":[{"delta":{"content":"ok"}}]}` + "\n"))
			if e != nil {
				t.Fatal(e)
			}
			var terminal [][]byte
			if end == "done" {
				more, _, err := sc.Feed([]byte("data: [DONE]\n"))
				terminal = more
				e = err
			} else {
				terminal, e = sc.FlushWithError()
			}
			all := append(events, terminal...)
			if e != nil || !strings.Contains(string(bytes.Join(all, nil)), "response.completed") {
				t.Fatalf("error=%v events=%s", e, bytes.Join(all, nil))
			}
		})
	}
	t.Run("empty stream fixed error", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		_, e := sc.FlushWithError()
		if e == nil || e.Message != "Responses stream ended without any response events" {
			t.Fatalf("error=%v", e)
		}
	})
}

func TestStreamConverterResponsesVariants(t *testing.T) {
	t.Run("choice-less usage chunk feeds response.completed", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		evs := feedAll(t, sc,
			`data: {"id":"r","choices":[{"delta":{"role":"assistant"}}]}`,
			`data: {"choices":[{"delta":{"content":"x"}}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4}}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`data: [DONE]`)
		var completed sseEvt
		for _, e := range evs {
			if e.Name == "response.completed" {
				completed = e
			}
		}
		resp := completed.Data["response"].(map[string]any)
		u, ok := resp["usage"].(map[string]any)
		if !ok || u["input_tokens"] != float64(9) || u["output_tokens"] != float64(4) || u["total_tokens"] != float64(13) {
			t.Fatalf("terminal event lost choice-less usage: %v", resp)
		}
	})
	t.Run("include_usage trailer after finish feeds response.completed", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		evs := feedAll(t, sc,
			`data: {"id":"r","choices":[{"delta":{"role":"assistant"}}]}`,
			`data: {"choices":[{"delta":{"content":"x"}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4}}`,
			`data: [DONE]`)
		var completed sseEvt
		for _, e := range evs {
			if e.Name == "response.completed" {
				completed = e
			}
		}
		resp := completed.Data["response"].(map[string]any)
		if resp["status"] != "completed" {
			t.Fatalf("status wrong: %v", resp)
		}
		output, ok := resp["output"].([]any)
		if !ok || len(output) != 1 || output[0].(map[string]any)["type"] != "message" {
			t.Fatalf("output items wrong: %v", resp["output"])
		}
		u, ok := resp["usage"].(map[string]any)
		if !ok || u["input_tokens"] != float64(9) || u["output_tokens"] != float64(4) || u["total_tokens"] != float64(13) {
			t.Fatalf("terminal event lost trailing include_usage usage: %v", resp)
		}
	})
	t.Run("length without usage maps to incomplete", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		evs := feedAll(t, sc,
			`data: {"id":"r","choices":[{"delta":{"role":"assistant"}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"length"}]}`,
			`data: [DONE]`)
		var completed sseEvt
		for _, e := range evs {
			if e.Name == "response.completed" {
				completed = e
			}
		}
		resp := completed.Data["response"].(map[string]any)
		if resp["status"] != "incomplete" {
			t.Fatalf("status wrong: %v", resp)
		}
		// Usage is always attached (F-R6); zero-valued when upstream sent none.
		u := resp["usage"].(map[string]any)
		if u["input_tokens"] != float64(0) || u["output_tokens"] != float64(0) || u["total_tokens"] != float64(0) {
			t.Fatalf("usage must default to zeros: %v", resp)
		}
	})
	t.Run("malformed chunk errors", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		_, _, eErr := sc.Feed([]byte("data: {bad\n"))
		if eErr == nil || eErr.Class != errclass.ClassTranslation {
			t.Fatalf("want ClassTranslation, got %+v", eErr)
		}
	})
	t.Run("in-stream error chunk fails both conversion targets", func(t *testing.T) {
		for _, source := range []string{"claude", "openai-response"} {
			sc := NewStreamConverter(source)
			events, done, eErr := sc.Feed([]byte("data: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n"))
			if done {
				t.Fatalf("%s: error chunk must not complete the stream", source)
			}
			if eErr == nil || eErr.Class != errclass.ClassUpstream || !eErr.Retryable {
				t.Fatalf("%s: want retryable upstream failure, got %+v", source, eErr)
			}
			if len(events) != 0 {
				t.Fatalf("%s: no partial success events may be emitted: %v", source, events)
			}
		}
	})
	t.Run("in-stream rate-limit error chunk classifies 429", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		_, _, eErr := sc.Feed([]byte("data: {\"error\":{\"message\":\"slow down\",\"code\":\"rate_limit_exceeded\"}}\n"))
		if eErr == nil || eErr.Class != errclass.ClassRateLimit || !eErr.Retryable || eErr.StatusCode != 429 {
			t.Fatalf("want retryable rate limit, got %+v", eErr)
		}
	})
	t.Run("non-data lines ignored", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		if evs, _, eErr := sc.Feed([]byte(": ping\n\n")); eErr != nil || len(evs) != 0 {
			t.Fatalf("comment lines must be ignored: %v %v", evs, eErr)
		}
	})
}

// Streamed tool calls + finish_reason:"length": the claude target's
// stop_reason keeps tool_use outranking max_tokens (TerminalReason),
// mirroring non-stream chatToClaude (FR-006). The Responses target derives
// status from finish alone (F-R2) — tools live in output items. No-tools
// streams keep length/max_tokens.
func TestStreamConverterTerminalReasonToolsOutrankLength(t *testing.T) {
	t.Run("claude stop_reason tool_use", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		evs := feedAll(t, sc,
			`data: {"id":"r","model":"m","choices":[{"delta":{"role":"assistant"}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{}"}}]}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"length"}]}`,
			`data: [DONE]`)
		var md sseEvt
		for _, e := range evs {
			if e.Name == "message_delta" {
				md = e
			}
		}
		if md.Data == nil || md.Data["delta"].(map[string]any)["stop_reason"] != "tool_use" {
			t.Fatalf("tool calls must outrank length: %v", md)
		}
	})
	t.Run("responses keeps finish-derived status, tools live in output", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		evs := feedAll(t, sc,
			`data: {"id":"r","choices":[{"delta":{"role":"assistant"}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{}"}}]}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"length"}]}`,
			`data: [DONE]`)
		var completed sseEvt
		for _, e := range evs {
			if e.Name == "response.completed" {
				completed = e
			}
		}
		resp := completed.Data["response"].(map[string]any)
		// F-R2: status derives from finish alone (length→incomplete); tool
		// calls are represented by output items, not status vocabulary.
		if resp["status"] != "incomplete" || len(resp["output"].([]any)) != 1 {
			t.Fatalf("length must map to incomplete with tool item in output: %v", resp)
		}
		item := resp["output"].([]any)[0].(map[string]any)
		if item["call_id"] != "c1" {
			t.Fatalf("terminal function_call missing call_id: %v", item)
		}
	})
	t.Run("no tools keeps status mapping unchanged", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		evs := feedAll(t, sc,
			`data: {"id":"r","model":"m","choices":[{"delta":{"content":"x"}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"length"}]}`,
			`data: [DONE]`)
		for _, e := range evs {
			if e.Name == "message_delta" && e.Data["delta"].(map[string]any)["stop_reason"] != "max_tokens" {
				t.Fatalf("no-tools length must stay max_tokens: %v", e)
			}
		}
	})
}

// Malformed-chunk errors redact bearer tokens from the echoed snippet and
// stay bounded (§5 security: never an upstream body echo).
func TestStreamConverterMalformedChunkRedactsBearer(t *testing.T) {
	sc := NewStreamConverter("claude")
	payload := `{"garbage":"prefix Bearer sk-secret123-token suffix ` + strings.Repeat("y", 200) + `"`
	_, _, eErr := sc.Feed([]byte("data: " + payload + "\n"))
	if eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Fatalf("want ClassTranslation, got %+v", eErr)
	}
	if strings.Contains(eErr.Message, "sk-secret123") {
		t.Fatalf("error leaks bearer token: %q", eErr.Message)
	}
	if len(eErr.Message) > 160 {
		t.Fatalf("error not bounded: %d chars", len(eErr.Message))
	}
}

// chunkError code classification (FR-009/§7): numeric JSON codes decode as
// float64, string codes may be numeric or rate-limit markers, and a
// rate-limit type upgrades an unclassified error to 429.
func TestChunkErrorCodeClassification(t *testing.T) {
	cases := []struct {
		name       string
		err        *ccChunkError
		wantClass  errclass.Class
		wantStatus int
	}{
		{"float64 numeric code", &ccChunkError{Message: "m", Code: float64(429)}, errclass.ClassRateLimit, 429},
		{"float64 server code", &ccChunkError{Message: "m", Code: float64(503)}, errclass.ClassUpstream, 503},
		{"string numeric code", &ccChunkError{Message: "m", Code: "500"}, errclass.ClassUpstream, 500},
		{"type rate limit no code", &ccChunkError{Message: "m", Type: "rate_limit_error"}, errclass.ClassRateLimit, 429},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := chunkError(tc.err)
			if got.Class != tc.wantClass || got.StatusCode != tc.wantStatus || !got.Retryable {
				t.Fatalf("want %s/%d retryable, got %+v", tc.wantClass, tc.wantStatus, got)
			}
		})
	}
}

// F-output pin: response.completed carries the output array — an
// aggregated output_text message for text-only streams (FR-006).
func TestStreamResponsesCompletedOutputText(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	evs := feedAll(t, sc,
		`data: {"id":"r9","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
		`data: {"choices":[{"delta":{"content":"Hel"}}]}`,
		`data: {"choices":[{"delta":{"content":"lo"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`,
		`data: [DONE]`)
	var completed map[string]any
	for _, e := range evs {
		if e.Name == "response.completed" {
			completed = e.Data["response"].(map[string]any)
		}
	}
	if completed == nil {
		t.Fatal("missing response.completed")
	}
	output, ok := completed["output"].([]any)
	if !ok || len(output) != 1 {
		t.Fatalf("output = %v", completed["output"])
	}
	item := output[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "assistant" || item["id"] != "msg_r9_0" || item["status"] != "completed" {
		t.Fatalf("message item wrong: %v", item)
	}
	content := item["content"].([]any)
	part := content[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "Hello" {
		t.Fatalf("output_text wrong: %v", content)
	}
}

// F-output pin: function_call items land in response.completed.output with
// their complete accumulated arguments.
func TestStreamResponsesCompletedOutputFunctionCall(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	evs := feedAll(t, sc,
		`data: {"id":"r8","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"cc","function":{"name":"f","arguments":""}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"j\":1,"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"k\":2}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		`data: [DONE]`)
	var completed map[string]any
	for _, e := range evs {
		t.Logf("EVENT %s", e)
		if e.Name == "response.completed" {
			completed = e.Data["response"].(map[string]any)
		}
	}
	if completed == nil {
		t.Fatal("missing response.completed")
	}
	output, ok := completed["output"].([]any)
	if !ok || len(output) != 1 {
		t.Fatalf("output = %v", completed["output"])
	}
	item := output[0].(map[string]any)
	// F-R5 pin: function_call items carry call_id only — no "id" key.
	if item["type"] != "function_call" || item["call_id"] != "cc" || item["name"] != "f" {
		t.Fatalf("function_call item wrong: %v", item)
	}
	if _, has := item["id"]; has {
		t.Fatalf("function_call must not carry an id key: %v", item)
	}
	if item["arguments"] != `{"j":1,"k":2}` {
		t.Fatalf("accumulated arguments wrong: %v", item["arguments"])
	}
}

// F1 pin (Messages-route invariant parity): when the upstream stream
// announces delta.tool_calls BEFORE delta.content — legal Chat Completions
// interleaving — response.completed.output must reproduce the streamed
// output_item.added indexes exactly: function_call at 0, message at 1,
// never displaced by unreserved text assembly.
func TestStreamResponsesToolsBeforeTextTerminalOrderMatchesAnnouncements(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	evs := feedAll(t, sc,
		`data: {"id":"rt","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":""}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`,
		`data: {"choices":[{"delta":{"content":"Hi"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		`data: [DONE]`)
	var added []sseEvt
	var completed map[string]any
	for _, e := range evs {
		switch e.Name {
		case "response.output_item.added":
			added = append(added, e)
		case "response.completed":
			completed = e.Data["response"].(map[string]any)
		}
	}
	if len(added) != 2 {
		t.Fatalf("want function_call + message announcements, got %v", added)
	}
	first := added[0].Data["item"].(map[string]any)
	if added[0].Data["output_index"] != float64(0) || first["type"] != "function_call" || first["call_id"] != "c1" {
		t.Fatalf("function_call must be announced at output_index 0: %v", added[0])
	}
	second := added[1].Data["item"].(map[string]any)
	if added[1].Data["output_index"] != float64(1) || second["type"] != "message" {
		t.Fatalf("message must be announced at output_index 1: %v", added[1])
	}
	output, ok := completed["output"].([]any)
	if !ok || len(output) != 2 {
		t.Fatalf("terminal output = %v", completed["output"])
	}
	termFC, ok := output[0].(map[string]any)
	if !ok || termFC["type"] != "function_call" || termFC["call_id"] != "c1" || termFC["arguments"] != "{}" {
		t.Fatalf("terminal output[0] must be the announced function_call: %v", output[0])
	}
	termMsg, ok := output[1].(map[string]any)
	if !ok || termMsg["type"] != "message" || termMsg["id"] != "msg_rt_1" {
		t.Fatalf("terminal output[1] must be the announced message: %v", output[1])
	}
	content := termMsg["content"].([]any)
	if content[0].(map[string]any)["text"] != "Hi" {
		t.Fatalf("aggregated text lost: %v", content)
	}
}

// An announced-but-zero-fragment tool call emits "{}" arguments in the
// terminal response.completed output like every sibling (DefaultArgs).
func TestStreamResponsesCompletedZeroFragmentArgs(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	evs := feedAll(t, sc,
		`data: {"id":"r7","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"cz","function":{"name":"f","arguments":""}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`)
	var completed map[string]any
	for _, e := range evs {
		if e.Name == "response.completed" {
			completed = e.Data["response"].(map[string]any)
		}
	}
	if completed == nil {
		t.Fatal("missing response.completed")
	}
	output, ok := completed["output"].([]any)
	if !ok || len(output) != 1 {
		t.Fatalf("output = %v", completed["output"])
	}
	if got := output[0].(map[string]any)["arguments"]; got != "{}" {
		t.Fatalf("zero-fragment arguments = %v, want {}", got)
	}
}

// F5 pin: upstream may legally close after the finish_reason chunk without
// [DONE]; Flush must deliver the deferred terminal exactly once — with the
// stranded usage (F6: input_tokens alongside output_tokens).
func TestStreamConverterFlushAfterFinishWithoutDONE(t *testing.T) {
	t.Run("claude target receives message_delta and message_stop once", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		if evs := feedAll(t, sc,
			`data: {"id":"r","model":"m","choices":[{"delta":{"role":"assistant","content":"x"}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}}`,
		); len(evs) == 0 {
			t.Fatal("finish chunk lost the block stops")
		}
		flushed := parseEvents(t, sc.Flush())
		if len(flushed) != 2 || flushed[0].Name != "message_delta" || flushed[1].Name != "message_stop" {
			t.Fatalf("flush = %v", flushed)
		}
		u, ok := flushed[0].Data["usage"].(map[string]any)
		if !ok || u["input_tokens"] != float64(7) || u["output_tokens"] != float64(5) {
			t.Fatalf("flushed message_delta usage wrong: %v", flushed[0])
		}
		if again := sc.Flush(); len(again) != 0 {
			t.Fatalf("second flush must return nothing: %v", again)
		}
	})
	t.Run("responses target receives response.completed", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		feedAll(t, sc,
			`data: {"id":"r2","choices":[{"delta":{"role":"assistant","content":"x"}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4}}`)
		flushed := parseEvents(t, sc.Flush())
		if len(flushed) != 4 || flushed[len(flushed)-1].Name != "response.completed" {
			t.Fatalf("flush = %v", flushed)
		}
		resp := flushed[len(flushed)-1].Data["response"].(map[string]any)
		u := resp["usage"].(map[string]any)
		if u["input_tokens"] != float64(3) || u["output_tokens"] != float64(4) {
			t.Fatalf("flushed completed usage wrong: %v", resp)
		}
		if again := sc.Flush(); len(again) != 0 {
			t.Fatalf("second flush must return nothing: %v", again)
		}
	})
	t.Run("no-op without a pending terminal or after DONE", func(t *testing.T) {
		passthrough := NewStreamConverter("openai")
		if evs := passthrough.Flush(); len(evs) != 0 {
			t.Fatalf("passthrough flush must be a no-op: %v", evs)
		}
		sc := NewStreamConverter("claude")
		feedAll(t, sc,
			`data: {"id":"r","model":"m","choices":[{"delta":{"role":"assistant"}}]}`,
			`data: [DONE]`)
		if evs := sc.Flush(); len(evs) != 0 {
			t.Fatalf("flush after [DONE] must return nothing: %v", evs)
		}
		noFinish := NewStreamConverter("claude")
		feedAll(t, noFinish, `data: {"id":"r","model":"m","choices":[{"delta":{"content":"x"}}]}`)
		if evs := noFinish.Flush(); len(evs) != 0 {
			t.Fatalf("flush without finish_reason must return nothing: %v", evs)
		}
	})
	t.Run("claude post-finish usage trailer then close-without-DONE still emits message_stop once", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		var deltaCount, stopCount int
		for _, ev := range feedAll(t, sc,
			`data: {"id":"r","model":"m","choices":[{"delta":{"content":"x"}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`data: {"id":"r","model":"m","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":5}}`,
		) {
			switch ev.Name {
			case "message_delta":
				deltaCount++
				u := ev.Data["usage"].(map[string]any)
				if u["input_tokens"] != float64(7) || u["output_tokens"] != float64(5) {
					t.Fatalf("trailer usage not carried into terminal: %v", ev)
				}
			case "message_stop":
				stopCount++
			}
		}
		if deltaCount != 1 || stopCount != 0 {
			t.Fatalf("trailer must deliver exactly one message_delta and no message_stop: delta=%d stop=%d", deltaCount, stopCount)
		}
		flushed := parseEvents(t, sc.Flush())
		if len(flushed) != 1 || flushed[0].Name != "message_stop" {
			t.Fatalf("flush after delivered terminal = %v, want lone message_stop", flushed)
		}
		if again := sc.Flush(); len(again) != 0 {
			t.Fatalf("second flush must return nothing: %v", again)
		}
	})
}
