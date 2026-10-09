package messages

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"ollama-cliproxyapi/internal/errclass"
)

func feed(t *testing.T, sc *StreamConverter, chunks ...string) ([][]byte, bool, *errclass.Error) {
	t.Helper()
	var events [][]byte
	done := false
	for _, c := range chunks {
		evs, d, e := sc.Feed([]byte(c))
		events = append(events, evs...)
		done = done || d
		if e != nil {
			return events, done, e
		}
	}
	return events, done, nil
}

// dataFrames extracts chat.completion.chunk payloads from openai events.
func dataFrames(t *testing.T, events [][]byte) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, e := range events {
		s := strings.TrimSpace(string(e))
		if s == "" || s == "[DONE]" || s == "data: [DONE]" {
			continue
		}
		s = strings.TrimPrefix(s, "data: ")
		var m map[string]any
		if err := json.Unmarshal([]byte(s), &m); err != nil {
			t.Fatalf("bad frame %q: %v", s, err)
		}
		out = append(out, m)
	}
	return out
}

// namedEvents indexes Responses-style events by their event: name.
func namedEvents(t *testing.T, events [][]byte) map[string][]map[string]any {
	t.Helper()
	out := map[string][]map[string]any{}
	for _, e := range events {
		s := string(e)
		if !strings.HasPrefix(s, "event: ") {
			continue
		}
		name, data, ok := strings.Cut(s, "\n")
		if !ok {
			t.Fatalf("unframed event %q", s)
		}
		var m map[string]any
		payload := strings.TrimPrefix(data, "data: ")
		if err := json.Unmarshal([]byte(strings.TrimSuffix(payload, "\n\n")), &m); err != nil {
			t.Fatalf("bad %s payload %q: %v", name, payload, err)
		}
		out[strings.TrimPrefix(name, "event: ")] = append(out[strings.TrimPrefix(name, "event: ")], m)
	}
	return out
}

func choice0(t *testing.T, frame map[string]any) map[string]any {
	t.Helper()
	return frame["choices"].([]any)[0].(map[string]any)
}

const claudeStream = "event: message_start\n" +
	"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"minimax\",\"usage\":{\"input_tokens\":10}}}\n" +
	"\n" +
	": keep-alive comment\n" +
	"\n" +
	"event: ping\n" +
	"data:{\"type\":\"ping\"}\n" +
	"\n" +
	"event: message_stop\n" +
	"data: {\"type\":\"message_stop\"}\n" +
	"\n"

func TestStreamClaudePassthroughVerbatim(t *testing.T) {
	sc := NewStreamConverter("claude")
	events, done, eErr := feed(t, sc, claudeStream)
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if !done {
		t.Error("message_stop must set done")
	}
	if len(events) != 3 { // message_start, ping, message_stop; comment blocks are inert
		t.Fatalf("events = %d", len(events))
	}
	if string(events[0]) != "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"minimax\",\"usage\":{\"input_tokens\":10}}}\n\n" {
		t.Errorf("verbatim lost: %q", events[0])
	}
	if !strings.Contains(string(events[1]), "data:{\"type\":\"ping\"}") {
		t.Errorf("ping not verbatim (spaceless data): %q", events[1])
	}
}

func TestStreamClaudeSplitAcrossFeedsAndCRLF(t *testing.T) {
	sc := NewStreamConverter("claude")
	// Split mid-token and use CRLF line endings throughout.
	stream := strings.ReplaceAll(claudeStream, "\n", "\r\n")
	mid := len(stream) / 2
	events, done, eErr := feed(t, sc, stream[:mid], stream[mid:])
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if !done {
		t.Error("done not reached")
	}
	joined := strings.Join(mapJoin(events), "")
	if !strings.Contains(joined, "event: message_start\r\n") {
		t.Errorf("CRLF not preserved: %q", joined)
	}
	if !strings.Contains(joined, "\"type\": \"message_stop\"") && !strings.Contains(joined, "\"type\":\"message_stop\"") {
		t.Errorf("final event missing: %q", joined)
	}
}

func mapJoin(events [][]byte) []string {
	out := make([]string, len(events))
	for i, e := range events {
		out[i] = string(e)
	}
	return out
}

func TestStreamClaudeErrorEvent(t *testing.T) {
	sc := NewStreamConverter("claude")
	_, _, eErr := feed(t, sc,
		"event: error\n",
		"data: {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow down\"}}\n\n",
	)
	if eErr == nil || eErr.Class != errclass.ClassRateLimit || !eErr.Retryable || eErr.StatusCode != 429 {
		t.Fatalf("want retryable rate_limit 429, got %+v", eErr)
	}
}

func TestStreamStrayBlankLineAndDataOnlyEvent(t *testing.T) {
	sc := NewStreamConverter("openai")
	events, done, eErr := feed(t, sc, "\n\n", "data: {\"type\":\"ping\"}\n\n")
	if eErr != nil || done || len(events) != 0 {
		t.Errorf("stray blank/data-only ping must be inert: %v %v %v", events, done, eErr)
	}
}

func TestStreamMalformedJSONRedactedSnippet(t *testing.T) {
	sc := NewStreamConverter("openai")
	_, _, eErr := feed(t, sc, "data: {oops \"t\":\"bearer sk-secret-value\"}\n\n")
	if eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Fatalf("want ClassTranslation, got %+v", eErr)
	}
	if strings.Contains(eErr.Message, "sk-secret-value") {
		t.Errorf("secret leaked: %q", eErr.Message)
	}
	if !strings.Contains(eErr.Message, "Bearer [redacted]") {
		t.Errorf("token not redacted: %q", eErr.Message)
	}

	long := "{oops" + strings.Repeat("A", 300)
	_, _, eErr = feed(t, NewStreamConverter("openai"), "data: "+long+"\n\n")
	if eErr == nil {
		t.Fatal("want error")
	}
	// Shared RedactedSnippet truncates to 80 payload chars + "...".
	if n := len(eErr.Message) - len("malformed SSE data JSON: "); n > 83 {
		t.Errorf("snippet too long: %d chars in %q", n, eErr.Message)
	}
	if !strings.HasSuffix(eErr.Message, "...") {
		t.Errorf("snippet missing ellipsis: %q", eErr.Message)
	}
}

func TestStreamUnknownFormat(t *testing.T) {
	sc := NewStreamConverter("grpc")
	_, _, eErr := feed(t, sc, "event: message_start\ndata: {}\n\n")
	if eErr == nil || eErr.Class != errclass.ClassUnsupported {
		t.Fatalf("want ClassUnsupported, got %+v", eErr)
	}
}

func TestStreamOpenAIFullFlow(t *testing.T) {
	sc := NewStreamConverter("openai")
	events, done, eErr := feed(t, sc,
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"minimax\",\"usage\":{\"input_tokens\":10}}}\n\n",
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hush\"}}\n\n",
		"event: content_block_stop\ndata: {\"index\":0}\n\n",
		"event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_a\",\"name\":\"f\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"1}\"}}\n\n",
		"event: content_block_start\ndata: {\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_b\",\"name\":\"g\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if !done {
		t.Error("done not reached")
	}
	joined := strings.Join(mapJoin(events), "")
	if strings.Contains(joined, "hush") {
		t.Error("thinking_delta leaked into openai stream (explicit-omission policy)")
	}
	frames := dataFrames(t, events)
	if len(frames) != 6 { // role, text, frag_a×2, frag_b, finish+usage
		t.Fatalf("frames = %d: %v", len(frames), frames)
	}
	first := choice0(t, frames[0])["delta"].(map[string]any)
	// Unified kernel shape: role chunks carry role plus empty content.
	if first["role"] != "assistant" || first["content"] != "" {
		t.Errorf("first chunk delta = %v", first)
	}
	if got := choice0(t, frames[1])["delta"].(map[string]any)["content"]; got != "hi" {
		t.Errorf("text delta = %v", got)
	}
	tc0 := choice0(t, frames[2])["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if tc0["index"] != float64(0) || tc0["id"] != "toolu_a" || tc0["type"] != "function" ||
		tc0["function"].(map[string]any)["name"] != "f" ||
		tc0["function"].(map[string]any)["arguments"] != "{\"a\":" {
		t.Errorf("first fragment = %v", tc0)
	}
	tc1 := choice0(t, frames[3])["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if _, hasID := tc1["id"]; hasID {
		t.Errorf("continuation fragment must omit id/name: %v", tc1)
	}
	if tc1["function"].(map[string]any)["arguments"] != "1}" {
		t.Errorf("continuation args = %v", tc1)
	}
	tcB := choice0(t, frames[4])["delta"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if tcB["index"] != float64(1) || tcB["id"] != "toolu_b" || tcB["type"] != "function" ||
		tcB["function"].(map[string]any)["name"] != "g" {
		t.Errorf("parallel tool block not separated by ordinal index: %v", tcB)
	}
	final := choice0(t, frames[5])
	if final["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v", final["finish_reason"])
	}
	usage := frames[5]["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(10) || usage["completion_tokens"] != float64(7) ||
		usage["total_tokens"] != float64(17) {
		t.Errorf("usage = %v", usage)
	}
	if frames[5]["object"] != "chat.completion.chunk" || frames[5]["model"] != "minimax" {
		t.Errorf("frame envelope = %v", frames[5])
	}
}

func TestStreamOpenAIUsageDetailsUseTerminalInputSnapshot(t *testing.T) {
	sc := NewStreamConverter("openai")
	events, _, eErr := feed(t, sc,
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"minimax\",\"usage\":{\"input_tokens\":538,\"cache_read_input_tokens\":0}}}\n\n",
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":410,\"output_tokens\":6,\"cache_read_input_tokens\":128,\"cache_creation_input_tokens\":0}}\n\n",
	)
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	frames := dataFrames(t, events)
	usage := frames[len(frames)-1]["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(538) || usage["completion_tokens"] != float64(6) ||
		usage["prompt_tokens_details"].(map[string]any)["cached_tokens"] != float64(128) {
		t.Fatalf("usage = %v", usage)
	}
}

func TestStreamOpenAIRoleFromFirstDelta(t *testing.T) {
	sc := NewStreamConverter("openai")
	events, _, eErr := feed(t, sc, "event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n")
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	frames := dataFrames(t, events)
	if len(frames) != 2 {
		t.Fatalf("frames = %d", len(frames))
	}
	d0 := choice0(t, frames[0])["delta"].(map[string]any)
	if d0["role"] != "assistant" || d0["content"] != "" {
		t.Errorf("role chunk missing before first delta (unified shape): %v", frames[0])
	}
}

func TestStreamOpenAIStopReasonMap(t *testing.T) {
	cases := []struct{ stop, want string }{
		{"end_turn", "stop"},
		{"stop_sequence", "stop"},
		{"tool_use", "tool_calls"},
		{"max_tokens", "length"},
		{"refusal", "content_filter"},
		{"something_new", "stop"},
	}
	for _, c := range cases {
		sc := NewStreamConverter("openai")
		events, _, eErr := feed(t, sc,
			"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\""+c.stop+"\"},\"usage\":{\"output_tokens\":1}}\n\n")
		if eErr != nil {
			t.Fatalf("%s: unexpected error: %v", c.stop, eErr)
		}
		got := choice0(t, dataFrames(t, events)[0])["finish_reason"]
		if got != c.want {
			t.Errorf("%s: finish_reason = %v, want %v", c.stop, got, c.want)
		}
	}
}

func TestStreamOpenAIMaxTokensWithToolsKeepsToolCalls(t *testing.T) {
	// Observed tool calls outrank the status-derived reason: a terminal
	// max_tokens must not downgrade finish_reason to length (FR-006).
	sc := NewStreamConverter("openai")
	events, _, eErr := feed(t, sc,
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"f\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":2}}\n\n",
	)
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	frames := dataFrames(t, events)
	if got := choice0(t, frames[len(frames)-1])["finish_reason"]; got != "tool_calls" {
		t.Errorf("finish_reason = %v, want tool_calls", got)
	}
}

func TestStreamOpenAIOrphanToolDelta(t *testing.T) {
	sc := NewStreamConverter("openai")
	_, _, eErr := feed(t, sc, "event: content_block_delta\ndata: {\"index\":5,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n")
	if eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Fatalf("want ClassTranslation, got %+v", eErr)
	}
	// Same failure when the block at the index is not a tool_use.
	sc = NewStreamConverter("openai")
	_, _, eErr = feed(t, sc,
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n")
	if eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Fatalf("non-tool block: want ClassTranslation, got %+v", eErr)
	}
}

func TestStreamOpenAIErrorEvent(t *testing.T) {
	sc := NewStreamConverter("openai")
	_, _, eErr := feed(t, sc, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"bad key\"}}\n\n")
	if eErr == nil || eErr.Class != errclass.ClassAuth || !eErr.Retryable || eErr.StatusCode != 401 {
		t.Fatalf("want retryable auth failure, got %+v", eErr)
	}
}

func TestStreamResponsesFlow(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	events, done, eErr := feed(t, sc,
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_9\",\"usage\":{\"input_tokens\":3}}}\n\n",
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hej\"}}\n\n",
		"event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"f\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"zz\"}}\n\n",
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if !done {
		t.Error("done not reached")
	}
	joined := strings.Join(mapJoin(events), "")
	if strings.Contains(joined, "zz") {
		t.Error("thinking_delta leaked into responses stream (explicit-omission policy)")
	}
	byName := namedEvents(t, events)
	created := byName["response.created"][0]
	createdResp := created["response"].(map[string]any)
	if createdResp["id"] != "msg_9" || createdResp["object"] != "response" || createdResp["status"] != "in_progress" {
		t.Errorf("created = %v", created)
	}
	if added := byName["response.output_item.added"]; len(added) != 2 {
		t.Errorf("want one output_item.added per block, got %v", added)
	} else {
		first := added[0]["item"].(map[string]any)
		second := added[1]["item"].(map[string]any)
		if first["type"] != "message" || first["id"] != "msg_9" ||
			second["type"] != "function_call" || second["call_id"] != "call_1" {
			t.Errorf("added items wrong: %v %v", first, second)
		}
		// function_call announcements stay call_id-only per the canonical
		// shape (deltas resolve item_id via call_id).
		if _, has := second["id"]; has {
			t.Errorf("function_call announcement must not carry an id key: %v", second)
		}
		if added[0]["output_index"] != float64(0) || added[1]["output_index"] != float64(1) {
			t.Errorf("added output_index wrong: %v %v", added[0], added[1])
		}
	}
	textDeltas := byName["response.output_text.delta"]
	if len(textDeltas) != 1 || textDeltas[0]["delta"] != "hej" || textDeltas[0]["item_id"] != "msg_9" ||
		textDeltas[0]["output_index"] != float64(0) {
		t.Errorf("text deltas = %v", textDeltas)
	}
	args := byName["response.function_call_arguments.delta"]
	if len(args) != 1 || args[0]["item_id"] != "call_1" || args[0]["output_index"] != float64(1) || args[0]["delta"] != "{}" {
		t.Errorf("args deltas = %v", args)
	}
	completed := byName["response.completed"][0]["response"].(map[string]any)
	usage := completed["usage"].(map[string]any)
	if usage["input_tokens"] != float64(3) || usage["output_tokens"] != float64(5) ||
		usage["total_tokens"] != float64(8) {
		t.Errorf("completed usage = %v", usage)
	}
	output, ok := completed["output"].([]any)
	if !ok || len(output) != 2 {
		t.Fatalf("completed output = %T %v", completed["output"], completed["output"])
	}
	msgItem := output[0].(map[string]any)
	if msgItem["type"] != "message" || msgItem["role"] != "assistant" ||
		msgItem["content"].([]any)[0].(map[string]any)["text"] != "hej" {
		t.Errorf("message item wrong: %v", msgItem)
	}
	// Terminal items match the non-stream converter shape exactly,
	// including the announced item identity (FR-006 sibling parity).
	if msgItem["id"] != "msg_9" {
		t.Errorf("terminal message item must carry the announcement id: %v", msgItem)
	}
	callItem := output[1].(map[string]any)
	if callItem["type"] != "function_call" || callItem["call_id"] != "call_1" ||
		callItem["name"] != "f" || callItem["arguments"] != "{}" {
		t.Errorf("function_call item wrong: %v", callItem)
	}
	if _, has := callItem["id"]; has {
		t.Errorf("function_call items stay call_id-only: %v", callItem)
	}
}

func TestStreamResponsesMaxTokensToolsIncomplete(t *testing.T) {
	// The Responses status vocabulary is {completed, incomplete}: a
	// max_tokens truncation maps to incomplete even when tool calls were
	// observed — those are represented by the function_call output item,
	// never by a "tool_calls" status override (FR-006; byte-shape parity
	// with the non-stream claudeToResponses converter).
	sc := NewStreamConverter("openai-response")
	events, done, eErr := feed(t, sc,
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{\"input_tokens\":1}}}\n\n",
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"c\",\"name\":\"f\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n",
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":9}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	completed := namedEvents(t, events)["response.completed"][0]["response"].(map[string]any)
	if got := completed["status"]; got != "incomplete" {
		t.Errorf("status = %v, want incomplete", got)
	}
	output, ok := completed["output"].([]any)
	if !ok || len(output) != 1 || output[0].(map[string]any)["call_id"] != "c" ||
		output[0].(map[string]any)["arguments"] != "{\"a\":1}" {
		t.Fatalf("truncated stream lost the accumulated tool call: %v", completed)
	}
}

// Terminal response.completed carries "object" and a stop_reason-derived
// "status" (max_tokens→incomplete, else completed), matching the Chat
// Completions route's synthesizer shape (FR-006).
func TestStreamResponsesTerminalStatusShape(t *testing.T) {
	t.Run("max_tokens maps to incomplete", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		events, done, eErr := feed(t, sc,
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"usage\":{\"input_tokens\":1}}}\n\n",
			"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":2}}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		)
		if eErr != nil || !done {
			t.Fatalf("done=%v err=%v", done, eErr)
		}
		resp := namedEvents(t, events)["response.completed"][0]["response"].(map[string]any)
		if resp["object"] != "response" || resp["status"] != "incomplete" {
			t.Fatalf("terminal shape = %v", resp)
		}
	})
	t.Run("end_turn maps to completed", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		events, done, eErr := feed(t, sc,
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m2\",\"usage\":{\"input_tokens\":1}}}\n\n",
			"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		)
		if eErr != nil || !done {
			t.Fatalf("done=%v err=%v", done, eErr)
		}
		resp := namedEvents(t, events)["response.completed"][0]["response"].(map[string]any)
		if resp["object"] != "response" || resp["status"] != "completed" {
			t.Fatalf("terminal shape = %v", resp)
		}
	})
}

func TestStreamResponsesOrphanToolDelta(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	_, _, eErr := feed(t, sc, "event: content_block_delta\ndata: {\"index\":9,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n")
	if eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Fatalf("want ClassTranslation, got %+v", eErr)
	}
}

func TestStreamResponsesErrorEvent(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	_, _, eErr := feed(t, sc, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"cap\"}}\n\n")
	if eErr == nil || eErr.Class != errclass.ClassUpstream || !eErr.Retryable {
		t.Fatalf("want retryable upstream failure, got %+v", eErr)
	}
}

func TestAnthropicErrorStatusTable(t *testing.T) {
	cases := map[string]int{
		"invalid_request_error":  400,
		"authentication_error":   401,
		"permission_error":       403,
		"not_found_error":        404,
		"request_too_large":      413,
		"rate_limit_error":       429,
		"overloaded_error":       503,
		"api_error":              500,
		"some_future_error_type": 500,
	}
	for typ, want := range cases {
		if got := anthropicErrorStatus(typ); got != want {
			t.Errorf("%s = %d, want %d", typ, got, want)
		}
	}
}

func TestStreamResponsesUnknownEventIgnored(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	events, done, eErr := feed(t, sc, "event: response.unknown_future\n"+`data: {"type":"response.unknown_future"}`+"\n\n")
	if eErr != nil || done || len(events) != 0 {
		t.Fatalf("unknown event must be inert: %v %v %v", events, done, eErr)
	}
}

// A claude-source frame without an event: line resolves its type from the
// payload's own "type" field (lazy probe in dispatchClaude).
func TestStreamClaudeTypeFallbackFromPayload(t *testing.T) {
	sc := NewStreamConverter("claude")
	events, done, eErr := feed(t, sc, `data: {"type":"message_stop"}`+"\n\n")
	if eErr != nil {
		t.Fatalf("fallback frame errored: %v", eErr)
	}
	// message_stop terminates the stream.
	if !done {
		t.Fatal("payload-typed message_stop must end the stream")
	}
	// Verbatim passthrough still emits the raw block.
	found := false
	for _, e := range events {
		if strings.Contains(string(e), `"type":"message_stop"`) {
			found = true
		}
	}
	if !found {
		t.Fatalf("passthrough block missing: %v", events)
	}
}

// Leading-thinking pin: FR-005 omits thinking blocks from the Responses
// output entirely, so streamed announcements/deltas must reference the
// SAME compacted positions response.completed.output renders — never raw
// upstream block indexes, which skip omitted blocks and point at items
// that do not exist in the terminal array.
func TestStreamResponsesLeadingThinkingCompactedOutputIndexes(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	events, done, eErr := feed(t, sc,
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_t\",\"usage\":{\"input_tokens\":1}}}\n\n",
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hush\"}}\n\n",
		"event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
		"event: content_block_start\ndata: {\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"f\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	if strings.Contains(strings.Join(mapJoin(events), ""), "hush") {
		t.Fatal("thinking leaked into responses stream (FR-005 omission policy)")
	}
	completed := namedEvents(t, events)["response.completed"][0]["response"].(map[string]any)
	output, ok := completed["output"].([]any)
	if !ok || len(output) != 2 {
		t.Fatalf("terminal output = %T %v", completed["output"], completed["output"])
	}
	idAt := make([]string, len(output)) // item identity per compacted position
	for i, it := range output {
		m := it.(map[string]any)
		switch m["type"] {
		case "message":
			idAt[i] = m["id"].(string)
		case "function_call":
			idAt[i] = m["call_id"].(string)
		default:
			t.Fatalf("unexpected terminal item %v", m)
		}
	}
	positionOf := func(id string) int {
		for i, got := range idAt {
			if got == id {
				return i
			}
		}
		t.Fatalf("item %q missing from terminal output %v", id, idAt)
		return -1
	}
	msgPos, callPos := positionOf("msg_t"), positionOf("call_1")
	if msgPos != 0 || callPos != 1 {
		t.Fatalf("compacted layout = %v", idAt)
	}
	byName := namedEvents(t, events)
	added := byName["response.output_item.added"]
	if len(added) != 2 ||
		added[0]["output_index"] != float64(msgPos) ||
		added[1]["output_index"] != float64(callPos) {
		t.Fatalf("announcements must reference terminal positions %d/%d: %v %v",
			msgPos, callPos, added[0], added[1])
	}
	if td := byName["response.output_text.delta"][0]; td["output_index"] != float64(msgPos) {
		t.Errorf("text delta output_index = %v, want %v", td["output_index"], msgPos)
	}
	if ad := byName["response.function_call_arguments.delta"][0]; ad["output_index"] != float64(callPos) {
		t.Errorf("args delta output_index = %v, want %v", ad["output_index"], callPos)
	}
}

// F5 pin: upstream closes after message_delta without message_stop; Flush
// must deliver the deferred terminal exactly once per target format.
func TestStreamConverterFlushAfterCloseWithoutMessageStop(t *testing.T) {
	feedToDelta := func(t *testing.T, sc *StreamConverter) {
		t.Helper()
		if _, done, eErr := feed(t, sc,
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_f\",\"model\":\"m\",\"usage\":{\"input_tokens\":3}}}\n\n",
			"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
			"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
			"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
		); eErr != nil || done {
			t.Fatalf("done=%v err=%v", done, eErr)
		}
	}

	t.Run("responses target receives response.completed exactly once", func(t *testing.T) {
		sc := NewStreamConverter("openai-response")
		feedToDelta(t, sc)
		evs := namedEvents(t, sc.Flush())
		if len(evs["response.completed"]) != 1 {
			t.Fatalf("flush delivered response.completed %d times", len(evs["response.completed"]))
		}
		resp := evs["response.completed"][0]["response"].(map[string]any)
		if resp["id"] != "msg_f" || resp["object"] != "response" || resp["status"] != "completed" {
			t.Errorf("completed = %v", resp)
		}
		u := resp["usage"].(map[string]any)
		if u["input_tokens"] != float64(3) || u["output_tokens"] != float64(2) || u["total_tokens"] != float64(5) {
			t.Errorf("usage = %v", u)
		}
		out, ok := resp["output"].([]any)
		if !ok || len(out) != 1 || out[0].(map[string]any)["type"] != "message" {
			t.Errorf("output = %v", resp["output"])
		}
		if again := sc.Flush(); len(again) != 0 {
			t.Fatalf("second flush must return nothing: %v", again)
		}
	})

	t.Run("openai target flush is empty", func(t *testing.T) {
		sc := NewStreamConverter("openai")
		feedToDelta(t, sc)
		flushed := sc.Flush()
		if len(flushed) != 0 {
			t.Fatalf("flush = %v", flushed)
		}
		if again := sc.Flush(); len(again) != 0 {
			t.Fatalf("second flush must return nothing: %v", again)
		}
	})

	t.Run("claude passthrough gets nothing extra", func(t *testing.T) {
		sc := NewStreamConverter("claude")
		_, done, eErr := feed(t, sc,
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"x\"}}\n\n",
			"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
		)
		if eErr != nil || done {
			t.Fatalf("done=%v err=%v", done, eErr)
		}
		if flushed := sc.Flush(); len(flushed) != 0 {
			t.Fatalf("passthrough flush must be empty: %v", flushed)
		}
	})

	t.Run("nothing emitted flushes nothing", func(t *testing.T) {
		for _, format := range []string{"openai-response", "openai"} {
			if flushed := NewStreamConverter(format).Flush(); len(flushed) != 0 {
				t.Fatalf("%s: empty-stream flush = %v", format, flushed)
			}
		}
	})
}

// W4 pin: [text, tool_use, text] must synthesize the canonical native
// shape — ONE message item aggregating every text block, positioned at
// the first text slot, with function_call items interleaved — and be
// byte-shape-equal to the non-stream claudeToResponses converter for the
// same upstream content (FR-006 mode parity).
func TestStreamResponsesTextAggregationMatchesNonStream(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	events, done, eErr := feed(t, sc,
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"usage\":{\"input_tokens\":3}}}\n\n",
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"A\"}}\n\n",
		"event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"c1\",\"name\":\"f\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n",
		"event: content_block_start\ndata: {\"index\":2,\"content_block\":{\"type\":\"text\"}}\n\n",
		"event: content_block_delta\ndata: {\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\"B\"}}\n\n",
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	if eErr != nil || !done {
		t.Fatalf("done=%v err=%v", done, eErr)
	}
	byName := namedEvents(t, events)
	added := byName["response.output_item.added"]
	if len(added) != 2 { // ONE message announcement + one function_call
		t.Fatalf("output_item.added count = %d: %v", len(added), added)
	}
	if added[0]["item"].(map[string]any)["type"] != "message" ||
		added[1]["item"].(map[string]any)["type"] != "function_call" {
		t.Errorf("announced items = %v %v", added[0], added[1])
	}
	deltas := byName["response.output_text.delta"]
	if len(deltas) != 2 || deltas[0]["delta"] != "A" || deltas[1]["delta"] != "B" {
		t.Fatalf("text deltas = %v", deltas)
	}
	// The second text block's delta targets the announced item's position.
	if deltas[1]["output_index"] != float64(0) || deltas[0]["output_index"] != float64(0) {
		t.Errorf("deltas must reference the single message item: %v", deltas)
	}

	completed := byName["response.completed"][0]["response"].(map[string]any)
	gotOutput := completed["output"]

	body := `{"id":"msg_x","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"text","text":"A"},{"type":"tool_use","id":"c1","name":"f","input":{"a":1}},` +
		`{"type":"text","text":"B"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":5}}`
	out, eErr := ConvertNonStreamResponse("openai-response", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("non-stream convert: %v", eErr)
	}
	var want map[string]any
	if err := json.Unmarshal(out, &want); err != nil {
		t.Fatalf("decode non-stream result: %v", err)
	}
	wantOutput := want["output"]
	var gotNorm, wantNorm any
	gotRaw, _ := json.Marshal(gotOutput)
	wantRaw, _ := json.Marshal(wantOutput)
	json.Unmarshal(gotRaw, &gotNorm)
	json.Unmarshal(wantRaw, &wantNorm)
	if !reflect.DeepEqual(gotNorm, wantNorm) {
		t.Fatalf("stream output diverges from non-stream:\n stream=%s\n non-stream=%s", gotRaw, wantRaw)
	}
	items := gotOutput.([]any)
	if len(items) != 2 {
		t.Fatalf("output items = %d: %v", len(items), items)
	}
	msgItem := items[0].(map[string]any)
	content := msgItem["content"].([]any)
	if msgItem["type"] != "message" || len(content) != 1 ||
		content[0].(map[string]any)["text"] != "AB" {
		t.Errorf("aggregated message item wrong: %v", msgItem)
	}
}
