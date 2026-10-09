package responses

import (
	"encoding/json"
	"strings"
	"testing"

	"ollama-cliproxyapi/internal/errclass"
)

// chunked splits a raw stream into small pieces so every test also
// exercises partial-line buffering across Feed calls.
func chunked(s string) []string {
	var out []string
	for len(s) > 0 {
		n := 6
		if n > len(s) {
			n = len(s)
		}
		out = append(out, s[:n])
		s = s[n:]
	}
	return out
}

func feedChunks(t *testing.T, sc *StreamConverter, chunks []string) (events [][]byte, done bool, eErr *errclass.Error) {
	t.Helper()
	for _, c := range chunks {
		evs, d, err := sc.Feed([]byte(c))
		events = append(events, evs...)
		done = done || d
		if err != nil {
			return events, done, err
		}
	}
	return events, done, nil
}

func runStream(t *testing.T, source, raw string) ([][]byte, bool, *errclass.Error) {
	t.Helper()
	return feedChunks(t, NewStreamConverter(source), chunked(raw))
}

func frame(event, data string) string {
	return "event: " + event + "\ndata: " + data + "\n\n"
}

func parseSSE(t *testing.T, b []byte) (event, data string) {
	t.Helper()
	var dataLines []string
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSuffix(line, "\r")
		switch {
		case strings.HasPrefix(line, "event:"):
			event = strings.TrimSpace(line[len("event:"):])
		case strings.HasPrefix(line, "data:"):
			dataLines = append(dataLines, strings.TrimSpace(line[len("data:"):]))
		}
	}
	return event, strings.Join(dataLines, "\n")
}

func payloadOf(t *testing.T, b []byte) map[string]any {
	t.Helper()
	s := strings.TrimSpace(string(b))
	var raw []byte
	if strings.HasPrefix(s, "data:") || strings.HasPrefix(s, "event:") {
		_, data := parseSSE(t, b)
		raw = []byte(data)
	} else {
		raw = []byte(s)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("bad synthesized payload %q: %v", raw, err)
	}
	return m
}

func choice0(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	choices, ok := m["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("missing choices: %v", m)
	}
	c, _ := choices[0].(map[string]any)
	return c
}

const createdPayload = `{"response":{"id":"resp_1","model":"gpt-5.6-luna","created_at":1700000000}}`

// ---- openai-response passthrough ----------------------------------------

func TestPassthroughVerbatimAndTerminal(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_text.delta", `{"delta":"hi"}`) +
		frame("response.completed", `{"response":{"usage":{"input_tokens":5,"output_tokens":7}}}`)
	events, done, eErr := runStream(t, "openai-response", raw)
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if len(events) != 3 || !done {
		t.Fatalf("events=%d done=%v", len(events), done)
	}
	want := frame("response.created", createdPayload)
	if string(events[0]) != want {
		t.Errorf("frame not verbatim:\n got %q\nwant %q", events[0], want)
	}
	if ev, _ := parseSSE(t, events[2]); ev != "response.completed" {
		t.Errorf("last event = %q", ev)
	}
}

func TestPassthroughIncompleteDone(t *testing.T) {
	_, done, eErr := runStream(t, "openai-response", frame("response.incomplete", `{}`))
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
}

func TestPassthroughFailedWithStatusCode(t *testing.T) {
	_, _, eErr := runStream(t, "openai-response",
		frame("response.failed", `{"status_code":429,"message":"slow down"}`))
	if eErr == nil || eErr.Class != errclass.ClassRateLimit || !eErr.Retryable || eErr.StatusCode != 429 {
		t.Fatalf("want rate limit via FromStatus, got %+v", eErr)
	}
}

func TestPassthroughErrorEventFallback(t *testing.T) {
	_, _, eErr := runStream(t, "openai-response", frame("error", `{"message":"boom"}`))
	if eErr == nil || eErr.Class != errclass.ClassUpstream || !eErr.Retryable || eErr.StatusCode != 0 {
		t.Fatalf("want retryable upstream failure, got %+v", eErr)
	}
}

func TestPassthroughDataOnlyFrame(t *testing.T) {
	events, done, eErr := runStream(t, "openai-response", "data: {\"x\":1}\n\n")
	if eErr != nil || done || len(events) != 1 {
		t.Fatalf("events=%d done=%v err=%v", len(events), done, eErr)
	}
	if strings.HasPrefix(string(events[0]), "event:") {
		t.Errorf("data-only frame must not gain an event line: %q", events[0])
	}
}

func TestPassthroughMultiLineDataJoined(t *testing.T) {
	// A legal multi-data-line frame must pass through BYTE-verbatim: a
	// rebuilt single "data:" line would embed raw newlines and corrupt
	// native Responses framing.
	raw := "event: response.output_text.delta\ndata: {\"delta\":\ndata: \"joined\"}\n\n"
	events, _, eErr := runStream(t, "openai-response", raw)
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if string(events[0]) != raw {
		t.Errorf("multi-line frame not byte-verbatim:\n got %q\nwant %q", events[0], raw)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte("{\"delta\":\n\"joined\"}"), &parsed); err != nil || parsed["delta"] != "joined" {
		t.Errorf("joined payload invalid: %v %v", parsed, err)
	}
}

func TestConversionMultiLineDataJoinedWithoutRaw(t *testing.T) {
	// Conversion targets run with wantRaw=false; a legal
	// multi-data-line frame must still join into one payload.
	raw := "event: response.output_text.delta\ndata: {\"delta\":\ndata: \"joined\"}\n\n" +
		frame("response.completed", `{"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if len(events) != 2 || !done {
		t.Fatalf("events=%d done=%v", len(events), done)
	}
	m := payloadOf(t, events[0])
	c := choice0(t, m)
	delta, _ := c["delta"].(map[string]any)
	if delta["content"] != "joined" {
		t.Errorf("joined delta lost: %v", delta)
	}
}

func TestUnknownSourceFormat(t *testing.T) {
	_, _, eErr := runStream(t, "grpc", frame("response.created", `{}`))
	if eErr == nil || eErr.Class != errclass.ClassUnsupported {
		t.Fatalf("want ClassUnsupported, got %+v", eErr)
	}
}

// Typed-decode pins: EVERY handled event type must fail translation on a
// malformed payload, and unknown-target rejection must apply per event
// type after successful decode (legacy precedence: malformed payload wins
// over ClassUnsupported).
func TestTypedDecodeErrorsOnEveryHandledEvent(t *testing.T) {
	cases := []struct{ event, payload string }{
		{"response.created", `{"response":{`},
		{"response.output_text.delta", `{"delta":`},
		{"response.output_item.added", `{"item":`},
		{"response.output_item.done", `{"item":`},
		{"response.function_call_arguments.delta", `{"item_id":"x",`},
		{"response.completed", `{"response":{`},
		{"response.incomplete", `{"response":{`},
		{"response.failed", `{`},
		{"error", `{`},
		// fallback path stays strict too
		{"response.unknown_future", `{bad`},
	}
	for _, c := range cases {
		_, _, eErr := runStream(t, "openai", frame(c.event, c.payload))
		if eErr == nil || eErr.Class != errclass.ClassTranslation {
			t.Errorf("%s: want ClassTranslation, got %+v", c.event, eErr)
		}
	}
}

// The unknown-event fallback validates payloads without decoding them
// into a discarded graph: malformed JSON still fails translation, while
// any valid payload — including non-object shapes — is inert.
func TestUnknownEventPayloadValidatedWithoutDecode(t *testing.T) {
	_, _, eErr := runStream(t, "openai", frame("response.unknown_future", `{bad`))
	if eErr == nil || eErr.Class != errclass.ClassTranslation ||
		!strings.Contains(eErr.Message, "malformed response.unknown_future event payload") {
		t.Fatalf("malformed unknown-event payload = %v", eErr)
	}
	for _, payload := range []string{``, `[1,2]`, `"note"`, `{"x":1}`} {
		events, done, eErr := runStream(t, "claude", frame("response.in_progress", payload))
		if eErr != nil || done || len(events) != 0 {
			t.Errorf("payload %q = %d events, done=%v, err=%v; want inert", payload, len(events), done, eErr)
		}
	}
}

func TestUnsupportedTargetRejectedOnEveryHandledEvent(t *testing.T) {
	cases := []struct{ event, payload string }{
		{"response.created", `{}`},
		{"response.output_text.delta", `{"delta":"x"}`},
		{"response.output_item.added", `{"item":{"type":"function_call"}}`},
		{"response.function_call_arguments.delta", `{"item_id":"x","delta":"y"}`},
		{"response.completed", `{}`},
		{"response.incomplete", `{}`},
		{"response.failed", `{}`},
		{"error", `{}`},
	}
	for _, c := range cases {
		_, _, eErr := runStream(t, "grpc", frame(c.event, c.payload))
		if eErr == nil || eErr.Class != errclass.ClassUnsupported {
			t.Errorf("%s: want ClassUnsupported, got %+v", c.event, eErr)
		}
	}
}

// Event-only frames (no data line) stay inert on conversion targets.
func TestConversionEventOnlyFrameInert(t *testing.T) {
	for _, source := range []string{"openai", "claude"} {
		events, done, eErr := runStream(t, source, "event: response.in_progress\n\n")
		if eErr != nil || done || len(events) != 0 {
			t.Errorf("%s: events=%d done=%v err=%v", source, len(events), done, eErr)
		}
	}
}

func TestStrayBlankLinesIgnored(t *testing.T) {
	events, done, eErr := runStream(t, "openai", "\n\n\n")
	if eErr != nil || done || len(events) != 0 {
		t.Fatalf("events=%d done=%v err=%v", len(events), done, eErr)
	}
}

func TestHeartbeatCommentAndEventOnlyFrame(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	events, _, eErr := feedChunks(t, sc, []string{": ping\n\n", "event: response.in_progress\n\n"})
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if len(events) != 1 {
		t.Fatalf("events=%d", len(events))
	}
	if ev, data := parseSSE(t, events[0]); ev != "response.in_progress" || data != "" {
		t.Errorf("event-only frame = %q / %q", ev, data)
	}
}

// ---- openai (Chat Completions) synthesis --------------------------------

func TestOpenAIRoleChunkAndTextDeltas(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_text.delta", `{"delta":"He"}`) +
		frame("response.output_text.delta", `{"delta":"y"}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	if len(events) != 3 {
		t.Fatalf("events=%d", len(events))
	}
	first := payloadOf(t, events[0])
	if first["object"] != "chat.completion.chunk" || first["id"] != "resp_1" ||
		first["model"] != "gpt-5.6-luna" || first["created"] != float64(1700000000) {
		t.Errorf("role chunk envelope = %v", first)
	}
	c := choice0(t, first)
	delta := c["delta"].(map[string]any)
	if delta["role"] != "assistant" || delta["content"] != "" {
		t.Errorf("role delta = %v", delta)
	}
	if c["finish_reason"] != nil {
		t.Errorf("intermediate finish_reason = %v", c["finish_reason"])
	}
	second := choice0(t, payloadOf(t, events[1]))
	if second["delta"].(map[string]any)["content"] != "He" {
		t.Errorf("text delta = %v", second["delta"])
	}
}

func TestOpenAIToolCallFlow(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":""}}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fc1","delta":"{\"q\":"}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fc1","delta":"\"x\"}"}`) +
		frame("response.completed", `{"response":{"usage":{"input_tokens":11,"output_tokens":3}}}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	if len(events) != 5 { // role, announce, 2 arg deltas, final
		t.Fatalf("events=%d: %s", len(events), events)
	}
	announce := choice0(t, payloadOf(t, events[1]))["delta"].(map[string]any)
	tcs := announce["tool_calls"].([]any)[0].(map[string]any)
	if tcs["index"] != float64(0) || tcs["id"] != "fc1" || tcs["type"] != "function" {
		t.Errorf("announce tool_call = %v", tcs)
	}
	if fn := tcs["function"].(map[string]any); fn["name"] != "lookup" || fn["arguments"] != "" {
		t.Errorf("announce function = %v", fn)
	}
	arg1 := choice0(t, payloadOf(t, events[2]))["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if arg1["index"] != float64(0) {
		t.Errorf("arg delta 1 index = %v", arg1["index"])
	}
	if _, hasID := arg1["id"]; hasID {
		t.Errorf("arg delta must extend the announced entry, not mint a new id: %v", arg1)
	}
	if arg1["function"].(map[string]any)["arguments"] != `{"q":` {
		t.Errorf("arg delta 1 = %v", arg1)
	}
	final := payloadOf(t, events[4])
	c := choice0(t, final)
	if c["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v", c["finish_reason"])
	}
	usage := final["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(11) || usage["completion_tokens"] != float64(3) || usage["total_tokens"] != float64(14) {
		t.Errorf("usage = %v", usage)
	}
}

func TestOpenAIToolCallReplayOnDone(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":""}}`) +
		frame("response.output_item.done", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":"{\"q\":1}"}}`) +
		frame("response.completed", `{}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	replay := choice0(t, payloadOf(t, events[2]))["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if replay["function"].(map[string]any)["arguments"] != `{"q":1}` {
		t.Errorf("args replay = %v", replay)
	}
	final := payloadOf(t, events[3])
	if choice0(t, final)["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v", final)
	}
	// F-R6: usage is always attached to the terminal chunk (zero-valued
	// when upstream sends none) for a stable schema.
	usage, has := final["usage"].(map[string]any)
	if !has || usage["prompt_tokens"] != float64(0) || usage["completion_tokens"] != float64(0) ||
		usage["total_tokens"] != float64(0) {
		t.Errorf("terminal chunk must always carry zero-valued usage when upstream sends none: %v", final)
	}
}

func TestOpenAISecondAddedIgnored(t *testing.T) {
	item := `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":""}}`
	events, _, eErr := runStream(t, "openai",
		frame("response.created", createdPayload)+frame("response.output_item.added", item)+frame("response.output_item.added", item))
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if len(events) != 2 {
		t.Fatalf("duplicate announce: events=%d", len(events))
	}
}

func TestOpenAINonFunctionItemIgnored(t *testing.T) {
	events, _, eErr := runStream(t, "openai",
		frame("response.output_item.done", `{"item":{"type":"message","content":[{"type":"output_text","text":"x"}]}}`))
	if eErr != nil || len(events) != 0 {
		t.Fatalf("events=%d err=%v", len(events), eErr)
	}
}

func TestOpenAICompletedStopNoTools(t *testing.T) {
	raw := frame("response.created", `{}`) +
		frame("response.output_text.delta", `{"delta":"hi"}`) +
		frame("response.completed", `{}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	if len(events) != 3 {
		t.Fatalf("events=%d", len(events))
	}
	start := payloadOf(t, events[0])
	if start["id"] != "commandcode" { // default identity when created carries no response object
		t.Errorf("default id = %v", start["id"])
	}
	// Upstream omitted created_at: the shared chunk kernel defaults to
	// render-time now instead of emitting a verbatim "created":0.
	if ts, ok := start["created"].(float64); !ok || ts <= 0 {
		t.Errorf("zero-created must default to render-time now, got %v", start["created"])
	}
	if got := choice0(t, payloadOf(t, events[2]))["finish_reason"]; got != "stop" {
		t.Errorf("finish_reason = %v", got)
	}
}

func TestOpenAIIncompleteLength(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.incomplete", `{"response":{"usage":{"input_tokens":2,"output_tokens":1}}}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	final := payloadOf(t, events[len(events)-1])
	if got := choice0(t, final)["finish_reason"]; got != "length" {
		t.Errorf("finish_reason = %v", got)
	}
}

func TestOpenAIIncompleteWithToolsKeepsToolCalls(t *testing.T) {
	// Shared precedence: response.incomplete cannot downgrade tool_calls.
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":""}}`) +
		frame("response.incomplete", `{}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	final := payloadOf(t, events[len(events)-1])
	if got := choice0(t, final)["finish_reason"]; got != "tool_calls" {
		t.Errorf("finish_reason = %v, want tool_calls", got)
	}
}

func TestOpenAIFailedMapsResponseStatus(t *testing.T) {
	_, _, eErr := runStream(t, "openai",
		frame("response.failed", `{"response":{"status_code":503,"error":{"message":"overloaded"}}}`))
	if eErr == nil || eErr.Class != errclass.ClassUpstream || !eErr.Retryable || eErr.StatusCode != 503 {
		t.Fatalf("want retryable upstream 503, got %+v", eErr)
	}
}

func TestOpenAIUnknownEventIgnored(t *testing.T) {
	events, _, eErr := runStream(t, "openai", frame("response.in_progress", `{"sequence_number":2}`))
	if eErr != nil || len(events) != 0 {
		t.Fatalf("events=%d err=%v", len(events), eErr)
	}
}

// ---- claude (Messages) synthesis ----------------------------------------

func messageStart(t *testing.T, events [][]byte) map[string]any {
	t.Helper()
	m := payloadOf(t, events[0])
	if m["type"] != "message_start" {
		t.Fatalf("first event = %v", m)
	}
	return m["message"].(map[string]any)
}

func TestClaudeMessageStartAndTextBlocks(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_text.delta", `{"delta":"Hello"}`) +
		frame("response.output_text.delta", `{"delta":" world"}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	if ev, _ := parseSSE(t, events[0]); ev != "message_start" {
		t.Errorf("first frame event name = %q (canonical frames are named)", ev)
	}
	msg := messageStart(t, events)
	if msg["id"] != "resp_1" || msg["model"] != "gpt-5.6-luna" || msg["role"] != "assistant" {
		t.Errorf("message_start message = %v", msg)
	}
	u := msg["usage"].(map[string]any)
	if u["input_tokens"] != float64(0) || u["output_tokens"] != float64(0) {
		t.Errorf("message_start usage = %v (canonical shape carries output_tokens:0)", u)
	}
	if len(events) != 4 { // start, block_start, delta, delta
		t.Fatalf("events=%d: %s", len(events), events)
	}
	start := payloadOf(t, events[1])
	if start["type"] != "content_block_start" || start["index"] != float64(0) {
		t.Errorf("block start = %v", start)
	}
	if cb := start["content_block"].(map[string]any); cb["type"] != "text" {
		t.Errorf("content_block = %v", cb)
	}
	d1 := payloadOf(t, events[2])
	if d1["type"] != "content_block_delta" || d1["index"] != float64(0) {
		t.Errorf("delta event = %v", d1)
	}
	if d := d1["delta"].(map[string]any); d["type"] != "text_delta" || d["text"] != "Hello" {
		t.Errorf("text_delta = %v", d)
	}
	d2 := payloadOf(t, events[3])
	if d2["delta"].(map[string]any)["text"] != " world" {
		t.Errorf("second text_delta = %v", d2)
	}
}

func TestClaudeToolUseFlow(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_text.delta", `{"delta":"hi"}`) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":""}}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fc1","delta":"{\"a\":1}"}`) +
		frame("response.completed", `{"response":{"usage":{"input_tokens":4,"output_tokens":9}}}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	// Mandated Anthropic pairing: message_start, text start/delta,
	// text STOP (before the tool block may start), tool_use start,
	// input_json_delta, tool stop, message_delta, message_stop.
	if len(events) != 9 {
		t.Fatalf("events=%d: %s", len(events), events)
	}
	stop0 := payloadOf(t, events[3])
	if stop0["type"] != "content_block_stop" || stop0["index"] != float64(0) {
		t.Fatalf("text block must stop before the tool_use start, got event 3 = %v", stop0)
	}
	toolStart := payloadOf(t, events[4])
	if toolStart["index"] != float64(1) {
		t.Errorf("tool block index = %v", toolStart["index"])
	}
	cb := toolStart["content_block"].(map[string]any)
	if cb["type"] != "tool_use" || cb["id"] != "fc1" || cb["name"] != "lookup" {
		t.Errorf("tool_use block = %v", cb)
	}
	argDelta := payloadOf(t, events[5])
	if argDelta["index"] != float64(1) {
		t.Errorf("arg delta index = %v", argDelta)
	}
	if d := argDelta["delta"].(map[string]any); d["type"] != "input_json_delta" || d["partial_json"] != `{"a":1}` {
		t.Errorf("input_json_delta = %v", d)
	}
	stop1 := payloadOf(t, events[6])
	if stop1["type"] != "content_block_stop" || stop1["index"] != float64(1) {
		t.Errorf("block stop 1 = %v", stop1)
	}
	md := payloadOf(t, events[7])
	if ev, _ := parseSSE(t, events[7]); ev != "message_delta" {
		t.Errorf("terminal delta event name = %q (canonical frames are named)", ev)
	}
	if md["type"] != "message_delta" {
		t.Fatalf("event 7 = %v", md)
	}
	d := md["delta"].(map[string]any)
	if d["stop_reason"] != "tool_use" {
		t.Errorf("message_delta delta = %v", d)
	}
	if _, ok := d["stop_sequence"]; ok {
		t.Errorf("stop_sequence must be omitted entirely, got %v", d)
	}
	mdUsage := md["usage"].(map[string]any)
	if mdUsage["output_tokens"] != float64(9) || mdUsage["input_tokens"] != float64(4) {
		t.Errorf("message_delta usage = %v (input tokens must ride along)", mdUsage)
	}
	stopEv, _ := parseSSE(t, events[8])
	if stopEv != "message_stop" || payloadOf(t, events[8])["type"] != "message_stop" {
		t.Errorf("event 8 = %q / %v", stopEv, payloadOf(t, events[8]))
	}
}

func TestClaudeToolDoneReplaysArgs(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":""}}`) +
		frame("response.output_item.done", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":"{\"k\":1}"}}`) +
		frame("response.output_item.done", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":""}}`) +
		frame("response.completed", `{}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	replay := payloadOf(t, events[2])
	if d := replay["delta"].(map[string]any); d["partial_json"] != `{"k":1}` {
		t.Errorf("args replay = %v", replay)
	}
	if len(events) != 6 { // empty-args done for a known block emits nothing
		t.Fatalf("events=%d: %s", len(events), events)
	}
	if got := payloadOf(t, events[4])["delta"].(map[string]any)["stop_reason"]; got != "tool_use" {
		t.Errorf("stop_reason = %v", got)
	}
}

func TestClaudeArgsDeltaBeforeItemAnnouncement(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.function_call_arguments.delta", `{"item_id":"fcX","delta":"{}"}`) +
		frame("response.completed", `{}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	start := payloadOf(t, events[1])
	cb := start["content_block"].(map[string]any)
	if cb["type"] != "tool_use" || cb["id"] != "fcX" || cb["name"] != "" {
		t.Errorf("defensive tool_use start = %v", cb)
	}
	if d := payloadOf(t, events[2])["delta"].(map[string]any); d["partial_json"] != "{}" {
		t.Errorf("defensive arg delta = %v", d)
	}
}

func TestClaudeToolDoneAsFirstEventStartsWithArgs(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.done", `{"item":{"type":"function_call","call_id":"fc2","name":"calc","arguments":"{\"n\":3}"}}`) +
		frame("response.completed", `{}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	start := payloadOf(t, events[1])
	cb := start["content_block"].(map[string]any)
	if cb["type"] != "tool_use" || cb["id"] != "fc2" || cb["name"] != "calc" {
		t.Errorf("tool_use start = %v", cb)
	}
	if d := payloadOf(t, events[2])["delta"].(map[string]any); d["partial_json"] != `{"n":3}` {
		t.Errorf("full args delta = %v", d)
	}
}

// Interleaved agentic turn: text delta, then a function_call item
// (start/delta/done), then more text. Anthropic Messages SSE mandates
// strict per-block pairing — every block must be closed before the next
// content_block_start, and deltas may only target open blocks. The
// synthesized stream must satisfy that invariant with stable incremental
// indices and reopen trailing text at a fresh index.
func TestClaudeInterleavedTextToolTextPairing(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_text.delta", `{"delta":"hi"}`) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":""}}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fc1","delta":"{\"a\":1}"}`) +
		frame("response.output_item.done", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":"{\"a\":1}"}}`) +
		frame("response.output_text.delta", `{"delta":"bye"}`) +
		frame("response.completed", `{"response":{"usage":{"input_tokens":3,"output_tokens":5}}}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}

	open := map[float64]bool{}
	text := map[float64]string{}
	jsonArgs := map[float64]string{}
	for i, ev := range events {
		m := payloadOf(t, ev)
		switch m["type"] {
		case "message_start":
			if i != 0 {
				t.Fatalf("message_start at %d", i)
			}
		case "message_delta", "message_stop":
		case "content_block_start":
			idx := m["index"].(float64)
			if len(open) != 0 {
				t.Fatalf("event %d: content_block_start(%v) while block(s) %v still open (close-before-next-start violated): %v", i, idx, open, m)
			}
			open[idx] = true
			cb := m["content_block"].(map[string]any)
			if cb["type"] == "tool_use" && (cb["id"] != "fc1" || cb["name"] != "lookup") {
				t.Errorf("event %d: tool_use block = %v", i, cb)
			}
		case "content_block_delta":
			idx := m["index"].(float64)
			if !open[idx] {
				t.Fatalf("event %d: content_block_delta(%v) on a closed/never-opened block: %v", i, idx, m)
			}
			d := m["delta"].(map[string]any)
			switch d["type"] {
			case "text_delta":
				text[idx] += d["text"].(string)
			case "input_json_delta":
				jsonArgs[idx] += d["partial_json"].(string)
			default:
				t.Fatalf("event %d: unexpected delta %v", i, d)
			}
		case "content_block_stop":
			idx := m["index"].(float64)
			if !open[idx] {
				t.Fatalf("event %d: content_block_stop(%v) on a closed/never-opened block: %v", i, idx, m)
			}
			delete(open, idx)
		default:
			t.Fatalf("event %d: unexpected type %v", i, m["type"])
		}
	}
	if len(open) != 0 {
		t.Fatalf("blocks left open at terminal: %v", open)
	}
	// Stable incremental indices: text block 0, its tool_use successor 1,
	// trailing reopened text at the next free index.
	if text[0] != "hi" || jsonArgs[1] != `{"a":1}` || text[2] != "bye" ||
		jsonArgs[0] != "" || jsonArgs[2] != "" || text[1] != "" {
		t.Fatalf("per-block content wrong: text=%v args=%v", text, jsonArgs)
	}
	md := payloadOf(t, events[len(events)-2])
	if d := md["delta"].(map[string]any); d["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", d)
	}
}

func TestClaudeNonFunctionItemIgnored(t *testing.T) {
	events, _, eErr := runStream(t, "claude",
		frame("response.output_item.added", `{"item":{"type":"reasoning","summary":[]}}`))
	if eErr != nil || len(events) != 0 {
		t.Fatalf("events=%d err=%v", len(events), eErr)
	}
}

func TestClaudeEndTurnNoTools(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_text.delta", `{"delta":"done"}`) +
		frame("response.completed", `{"response":{"usage":{"input_tokens":1,"output_tokens":2}}}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	if len(events) != 6 { // start, block_start, delta, block_stop, message_delta, message_stop
		t.Fatalf("events=%d: %s", len(events), events)
	}
	md := payloadOf(t, events[4])
	if d := md["delta"].(map[string]any); d["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v", d)
	}
	endUsage := md["usage"].(map[string]any)
	if endUsage["output_tokens"] != float64(2) || endUsage["input_tokens"] != float64(1) {
		t.Errorf("usage = %v", endUsage)
	}
	if payloadOf(t, events[5])["type"] != "message_stop" {
		t.Errorf("event 5 = %v", payloadOf(t, events[5]))
	}
}

func TestClaudeIncompleteMaxTokens(t *testing.T) {
	raw := frame("response.created", `{}`) + frame("response.incomplete", `{"response":{}}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	msg := messageStart(t, events)
	if msg["id"] != "commandcode" { // default identity preserved
		t.Errorf("default id = %v", msg["id"])
	}
	md := payloadOf(t, events[len(events)-2])
	if d := md["delta"].(map[string]any); d["stop_reason"] != "max_tokens" {
		t.Errorf("stop_reason = %v", d)
	}
	if md["usage"].(map[string]any)["output_tokens"] != float64(0) {
		t.Errorf("usage = %v", md["usage"])
	}
}

func TestClaudeIncompleteWithToolsKeepsToolUse(t *testing.T) {
	// Same precedence in Messages vocabulary: incomplete cannot
	// downgrade tool_use to max_tokens.
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fc1","name":"lookup","arguments":""}}`) +
		frame("response.incomplete", `{}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	md := payloadOf(t, events[len(events)-2])
	if d := md["delta"].(map[string]any); d["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", d)
	}
}

func TestClaudeErrorEventClassified(t *testing.T) {
	_, _, eErr := runStream(t, "claude", frame("error", `{"message":"upstream exploded"}`))
	if eErr == nil || eErr.Class != errclass.ClassUpstream || !eErr.Retryable {
		t.Fatalf("want retryable upstream failure, got %+v", eErr)
	}
}

func TestClaudeUnknownEventIgnored(t *testing.T) {
	events, _, eErr := runStream(t, "claude", frame("response.in_progress", `{}`))
	if eErr != nil || len(events) != 0 {
		t.Fatalf("events=%d err=%v", len(events), eErr)
	}
}

// ---- framing robustness --------------------------------------------------

func TestPartialLineSplitAcrossFeeds(t *testing.T) {
	sc := NewStreamConverter("openai")
	parts := []string{"even", "t: response.created\nda", "ta: {\"response\":{\"id\":\"r\"}}\n", "\n"}
	events, done, eErr := feedChunks(t, sc, parts)
	if eErr != nil || done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	if len(events) != 1 {
		t.Fatalf("events=%d", len(events))
	}
	if id := payloadOf(t, events[0])["id"]; id != "r" {
		t.Errorf("split frame lost identity: %v", id)
	}
}

func TestCRLFStreamTolerated(t *testing.T) {
	raw := "event: response.created\r\ndata: " + createdPayload + "\r\n\r\n" +
		"event: response.completed\r\ndata: {}\r\n\r\n"
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	if len(events) != 2 { // role chunk + final
		t.Fatalf("events=%d: %s", len(events), events)
	}
	if id := payloadOf(t, events[0])["id"]; id != "resp_1" {
		t.Errorf("CRLF frame lost identity: %v", id)
	}
}

func TestMalformedPayloadTranslationError(t *testing.T) {
	longBad := `{"delta":"` + strings.Repeat("x", 100)
	sc := NewStreamConverter("openai")
	events, _, eErr := feedChunks(t, sc, []string{
		frame("response.created", createdPayload),
		frame("response.output_text.delta", longBad),
	})
	if eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Fatalf("want ClassTranslation, got %+v", eErr)
	}
	if len(events) != 1 { // events before the failure are still returned
		t.Fatalf("events=%d", len(events))
	}
	if !strings.Contains(eErr.Message, "malformed response.output_text.delta event payload") {
		t.Errorf("message = %q", eErr.Message)
	}
	if !strings.HasSuffix(eErr.Message, "...") || len(eErr.Message) > 200 {
		t.Errorf("snippet not truncated/redacted: %q", eErr.Message)
	}
	short, _, eErr := runStream(t, "openai", frame("response.output_text.delta", `{bad`))
	if eErr == nil || !strings.HasSuffix(eErr.Message, "{bad") {
		t.Fatalf("short snippet = %+v (events %d)", eErr, len(short))
	}
}

// F13 fix pin: two interleaved function_call items must announce
// separately with distinct incremental indexes, keep their argument
// deltas on their own entries, and support per-call done replay.
func TestOpenAITwoInterleavedToolCalls(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fa","name":"fa","arguments":""}}`) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fb","name":"fb","arguments":""}}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fa","delta":"{\"a\":"}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fb","delta":"{\"b\":"}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fa","delta":"1}"}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fb","delta":"2}"}`) +
		frame("response.completed", `{"response":{"usage":{"input_tokens":1,"output_tokens":2}}}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}

	type tcEntry struct {
		index float64
		id    string
		name  string
		args  string
	}
	var calls []tcEntry
	var finish string
	for _, ev := range events {
		if strings.HasPrefix(string(ev), "data: [DONE]") {
			continue // terminal sentinel, not JSON
		}
		payload := payloadOf(t, ev)
		if _, has := payload["choices"]; !has {
			continue // data: [DONE]
		}
		c := choice0(t, payload)
		if fr, ok := c["finish_reason"].(string); ok && fr != "" {
			finish = fr
		}
		deltaMap := c["delta"].(map[string]any)
		tcs, hasTCs := deltaMap["tool_calls"].([]any)
		if !hasTCs {
			continue // role/text/terminal chunks carry no tool_calls
		}
		for _, tc := range tcs {
			e := tc.(map[string]any)
			entry := tcEntry{index: e["index"].(float64)}
			if id, ok := e["id"].(string); ok {
				entry.id = id // announce/delta-open entries carry the id
			}
			if fn, ok := e["function"].(map[string]any); ok {
				if n, ok := fn["name"].(string); ok {
					entry.name = n // only announcement entries carry the name
				}
				entry.args = fn["arguments"].(string)
			}
			calls = append(calls, entry)
		}
		if fr, ok := c["finish_reason"].(string); ok && fr != "" {
			finish = fr
		}
	}
	if finish != "tool_calls" {
		t.Fatalf("finish_reason = %q", finish)
	}
	// Announces: fa idx0 (no args yet), fb idx1 (no args yet).
	if len(calls) < 2 || calls[0].index != 0 || calls[0].id != "fa" || calls[0].args != "" ||
		calls[1].index != 1 || calls[1].id != "fb" {
		t.Fatalf("announces wrong: %+v", calls)
	}
	// Argument deltas land on their OWN entries.
	deltas := calls[2:]
	want := []struct {
		index float64
		args  string
	}{{0, `{"a":`}, {1, `{"b":`}, {0, `1}`}, {1, `2}`}}
	for i, w := range want {
		if deltas[i].index != w.index || deltas[i].args != w.args {
			t.Fatalf("delta %d = %+v, want index %v args %s", i, deltas[i], w.index, w.args)
		}
	}
}

// Done-replay is per call_id: a second call's done event replays its own
// arguments without resurrecting the first call's.
func TestOpenAIToolDoneReplayPerCall(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"f1","name":"one","arguments":""}}`) +
		frame("response.output_item.done", `{"item":{"type":"function_call","call_id":"f1","name":"one","arguments":"{\"x\":1}"}}`) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"f2","name":"two","arguments":""}}`) +
		frame("response.output_item.done", `{"item":{"type":"function_call","call_id":"f2","name":"two","arguments":"{\"y\":2}"}}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	var replays []struct {
		index float64
		args  string
	}
	for _, ev := range events {
		if strings.HasPrefix(string(ev), "data: [DONE]") {
			continue // terminal sentinel, not JSON
		}
		payload := payloadOf(t, ev)
		if _, has := payload["choices"]; !has {
			continue // data: [DONE]
		}
		c := choice0(t, payload)
		deltaMap := c["delta"].(map[string]any)
		tcs, hasTCs := deltaMap["tool_calls"].([]any)
		if !hasTCs {
			continue // role/text/terminal chunks carry no tool_calls
		}
		for _, tc := range tcs {
			e := tc.(map[string]any)
			replays = append(replays, struct {
				index float64
				args  string
			}{e["index"].(float64), e["function"].(map[string]any)["arguments"].(string)})
		}
	}
	// f1 announce(no args) + f1 replay + f2 announce(no args) + f2 replay.
	want := []struct {
		index float64
		args  string
	}{
		{0, ""},
		{0, `{"x":1}`},
		{1, ""},
		{1, `{"y":2}`},
	}
	if len(replays) != len(want) {
		t.Fatalf("replays = %+v", replays)
	}
	for i, w := range want {
		if replays[i].index != w.index || replays[i].args != w.args {
			t.Fatalf("entry %d = %+v, want %+v", i, replays[i], w)
		}
	}
}

// Announce-with-arguments: an upstream that streams complete arguments in
// the added frame emits one entry carrying them, with no later replay.
func TestOpenAIToolAnnounceCarriesArguments(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fcz","name":"zf","arguments":"{\"z\":9}"}}`) +
		frame("response.completed", `{"response":{"status":"completed"}}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	var entries []struct {
		index float64
		args  string
	}
	for _, ev := range events {
		if strings.HasPrefix(string(ev), "data: [DONE]") {
			continue
		}
		c := choice0(t, payloadOf(t, ev))
		deltaMap := c["delta"].(map[string]any)
		if tcs, ok := deltaMap["tool_calls"].([]any); ok {
			for _, tc := range tcs {
				e := tc.(map[string]any)
				fn := e["function"].(map[string]any)
				entries = append(entries, struct {
					index float64
					args  string
				}{e["index"].(float64), fn["arguments"].(string)})
			}
		}
	}
	if len(entries) != 1 || entries[0].index != 0 || entries[0].args != `{"z":9}` {
		t.Fatalf("announce entries = %+v", entries)
	}
}

// Dual-key pin: arguments streamed BEFORE the item announcement open one
// entry keyed by item_id; the later added must resolve that state through
// the alias (call_id) instead of forking a ghost entry, and the
// continuation delta extends the same index.
func TestOpenAIArgsDeltaBeforeAnnouncementIsOneCall(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.function_call_arguments.delta", `{"item_id":"fc_1","delta":"{\"a\":"}`) +
		frame("response.output_item.added", `{"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"calc","arguments":""}}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fc_1","delta":"1}"}`) +
		frame("response.completed", `{}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}

	type tcEntry struct {
		index float64
		id    string
		args  string
	}
	var calls []tcEntry
	finish := ""
	for _, ev := range events {
		if strings.HasPrefix(string(ev), "data: [DONE]") {
			continue // terminal sentinel, not JSON
		}
		c := choice0(t, payloadOf(t, ev))
		if fr, ok := c["finish_reason"].(string); ok && fr != "" {
			finish = fr
			continue
		}
		deltaMap := c["delta"].(map[string]any)
		tcs, ok := deltaMap["tool_calls"].([]any)
		if !ok {
			continue
		}
		for _, tc := range tcs {
			e := tc.(map[string]any)
			entry := tcEntry{index: e["index"].(float64)}
			if id, ok := e["id"].(string); ok {
				entry.id = id
			}
			entry.args = e["function"].(map[string]any)["arguments"].(string)
			calls = append(calls, entry)
		}
	}
	// Exactly: delta-open, continuation — the announcement emits nothing.
	if len(calls) != 2 || calls[0].index != 0 || calls[0].id != "fc_1" || calls[0].args != `{"a":` ||
		calls[1].index != 0 || calls[1].args != `1}` {
		t.Fatalf("tool entries wrong (ghost or split): %+v", calls)
	}
	if finish != "tool_calls" {
		t.Fatalf("finish_reason = %q", finish)
	}
}

// Divergence-fix pin: output_item.added carrying COMPLETE arguments must
// deliver identical, non-empty tool arguments through BOTH conversion
// targets — including when upstream closes WITHOUT a done event (the
// Messages route used to emit an empty-input tool_use block) — and a
// later done repeating them must not duplicate the delivery.
func TestAddedCompleteArgsParityAcrossTargets(t *testing.T) {
	added := frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"fc9","name":"zap","arguments":"{\"k\":7}"}}`)
	scenarios := []struct {
		name string
		raw  string
	}{
		{"no done event", added},
		{"done repeats the args", added + frame("response.output_item.done",
			`{"item":{"type":"function_call","call_id":"fc9","name":"zap","arguments":"{\"k\":7}"}}`)},
	}
	for _, s := range scenarios {
		t.Run(s.name, func(t *testing.T) {
			raw := frame("response.created", createdPayload) + s.raw + frame("response.completed", `{}`)
			const want = `{"k":7}`
			ccEvents, done, eErr := runStream(t, "openai", raw)
			if eErr != nil || !done {
				t.Fatalf("openai: done=%v err=%v", done, eErr)
			}
			clEvents, done, eErr := runStream(t, "claude", raw)
			if eErr != nil || !done {
				t.Fatalf("claude: done=%v err=%v", done, eErr)
			}
			ccArgs, clArgs := "", ""
			for _, ev := range ccEvents {
				if strings.HasPrefix(string(ev), "data: [DONE]") {
					continue // terminal sentinel, not JSON
				}
				m := payloadOf(t, ev)
				if _, has := m["choices"]; !has {
					continue // data: [DONE]
				}
				delta := choice0(t, m)["delta"].(map[string]any)
				if tcs, ok := delta["tool_calls"].([]any); ok {
					for _, tc := range tcs {
						ccArgs += tc.(map[string]any)["function"].(map[string]any)["arguments"].(string)
					}
				}
			}
			for _, ev := range clEvents {
				if strings.HasPrefix(string(ev), "data: [DONE]") {
					continue // terminal sentinel, not JSON
				}
				d, ok := payloadOf(t, ev)["delta"].(map[string]any)
				if ok && d["type"] == "input_json_delta" {
					clArgs += d["partial_json"].(string)
				}
			}
			if ccArgs != want || clArgs != want {
				t.Fatalf("divergent tool arguments: chat-completions=%q messages=%q, both want %q", ccArgs, clArgs, want)
			}
		})
	}
}

// Namespace-alias pin: OpenAI-conformant upstreams announce a function
// call under call_id ("call_1") while argument deltas reference the
// item's own id ("fc_1"). Both must resolve to ONE tracker state — one
// announced entry, deltas extending it at its index, and no ghost entry
// beside it (which used to duplicate args under two call ids at replay).
func TestOpenAICallIDAndItemIDAreOneCall(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"call_1","id":"fc_1","name":"lookup","arguments":""}}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fc_1","delta":"{\"q\":"}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fc_1","delta":"\"x\"}"}`) +
		frame("response.completed", `{"response":{"usage":{"input_tokens":1,"output_tokens":2}}}`)
	events, done, eErr := runStream(t, "openai", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	if len(events) != 5 { // role, announce, 2 arg deltas, final — no ghost events
		t.Fatalf("events=%d: %s", len(events), events)
	}
	announce := choice0(t, payloadOf(t, events[1]))["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if announce["index"] != float64(0) || announce["id"] != "call_1" || announce["type"] != "function" {
		t.Errorf("announce = %v", announce)
	}
	for i, w := range []string{`{"q":`, `"x"}`} {
		e := choice0(t, payloadOf(t, events[2+i]))["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
		if e["index"] != float64(0) {
			t.Errorf("delta %d index = %v, want 0 (announced entry)", i, e["index"])
		}
		if _, hasID := e["id"]; hasID {
			t.Errorf("delta %d must extend the announced entry, not open a second id: %v", i, e)
		}
		if got := e["function"].(map[string]any)["arguments"]; got != w {
			t.Errorf("delta %d args = %v, want %s", i, got, w)
		}
	}
}

// Messages-vocabulary twin of the namespace-alias pin: deltas carrying
// item_id "fc_1" feed the block announced under call_id "call_1" — one
// tool_use block, deltas at its index, done replay suppressed as already
// delivered, and no nameless/empty-input ghost block.
func TestClaudeCallIDAndItemIDAreOneBlock(t *testing.T) {
	raw := frame("response.created", createdPayload) +
		frame("response.output_text.delta", `{"delta":"hi"}`) +
		frame("response.output_item.added", `{"item":{"type":"function_call","call_id":"call_1","id":"fc_1","name":"lookup","arguments":""}}`) +
		frame("response.function_call_arguments.delta", `{"item_id":"fc_1","delta":"{\"a\":1}"}`) +
		frame("response.output_item.done", `{"item":{"type":"function_call","call_id":"call_1","id":"fc_1","name":"lookup","arguments":"{\"a\":1}"}}`) +
		frame("response.completed", `{}`)
	events, done, eErr := runStream(t, "claude", raw)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	// start, text start, text delta, text stop (paired before tool start),
	// tool_use start, input_json_delta,
	// stop x2 total, message_delta, message_stop — a ghost block would add more.
	if len(events) != 9 {
		t.Fatalf("events=%d: %s", len(events), events)
	}
	starts, stops := 0, 0
	var stopIndexes []any
	for _, ev := range events {
		m := payloadOf(t, ev)
		switch m["type"] {
		case "content_block_start":
			if cb := m["content_block"].(map[string]any); cb["type"] == "tool_use" {
				starts++
				if cb["id"] != "call_1" || cb["name"] != "lookup" || len(cb["input"].(map[string]any)) != 0 {
					t.Errorf("tool_use block = %v", cb)
				}
			}
		case "content_block_stop":
			stops++
			stopIndexes = append(stopIndexes, m["index"])
		}
	}
	if starts != 1 {
		t.Errorf("tool_use content_block_start count = %d, want 1", starts)
	}
	if stops != 2 {
		t.Errorf("content_block_stop count = %d (%v), want exactly text+tool blocks", stops, stopIndexes)
	}
	argDelta := payloadOf(t, events[5])
	if argDelta["index"] != float64(1) {
		t.Errorf("arg delta index = %v, want 1 (announced block)", argDelta["index"])
	}
	if d := argDelta["delta"].(map[string]any); d["partial_json"] != `{"a":1}` {
		t.Errorf("arg delta = %v", d)
	}
}
