package shared

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"ollama-cliproxyapi/internal/errclass"
)

func TestRewriteModelID(t *testing.T) {
	out, eErr := RewriteModelID("glm-5.2", []byte(`{"model":"whatever","messages":[{"role":"user"}]}`), "openai")
	if eErr != nil {
		t.Fatalf("success case: %v", eErr)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("output not json: %v", err)
	}
	if got["model"] != "glm-5.2" {
		t.Fatalf("model = %v", got["model"])
	}
	msgs, _ := got["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("other fields not preserved: %v", got)
	}

	for name, body := range map[string]string{
		"malformed":  "{bad",
		"non-object": "[1]",
		"scalar":     `"just a string"`,
	} {
		_, eErr := RewriteModelID("m", []byte(body), "claude")
		if eErr == nil || eErr.Class != errclass.ClassTranslation {
			t.Fatalf("%s: err = %v, want translation failure", name, eErr)
		}
		if !strings.Contains(eErr.Message, "malformed claude request JSON") {
			t.Fatalf("%s: message missing label: %q", name, eErr.Message)
		}
	}
}

// F4 regression PoC: a literal JSON null body decodes into a NIL map
// (encoding/json leaves the destination untouched for top-level null),
// so the model assignment below used to panic with "assignment to entry
// in nil map". JSON null is a malformed request body, not a crash.
func TestRewriteModelIDJSONNull(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("JSON null body panicked: %v", r)
		}
	}()
	out, eErr := RewriteModelID("m", []byte(`null`), "openai")
	if eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Fatalf("JSON null body = %q, %v; want translation failure", out, eErr)
	}
	if !strings.Contains(eErr.Message, "malformed") || !strings.Contains(eErr.Message, "JSON null") {
		t.Fatalf("message = %q", eErr.Message)
	}
}

// The single-decode rewrite path must also handle an ABSENT or null model
// field (no fast-path match) by writing the upstream ID like any other
// differing value.
func TestRewriteModelIDAbsentOrNullModel(t *testing.T) {
	for name, body := range map[string]string{
		"absent": `{"messages":[]}`,
		"null":   `{"model":null,"messages":[]}`,
	} {
		out, eErr := RewriteModelID("glm-5.2", []byte(body), "openai")
		if eErr != nil {
			t.Fatalf("%s: %v", name, eErr)
		}
		var got map[string]any
		if err := json.Unmarshal(out, &got); err != nil || got["model"] != "glm-5.2" {
			t.Fatalf("%s: model = %v (%v)", name, got["model"], err)
		}
	}
}

func TestHasContent(t *testing.T) {
	for name, raw := range map[string]json.RawMessage{
		"nil":   nil,
		"empty": {},
		"null":  json.RawMessage(`null`),
	} {
		if HasContent(raw) {
			t.Errorf("%s: HasContent = true, want false (%s)", name, raw)
		}
	}
	if !HasContent(json.RawMessage(`""`)) || !HasContent(json.RawMessage(`{}`)) {
		t.Error("populated raw rejected")
	}
}

func TestDecodeArgs(t *testing.T) {
	for _, in := range []string{"", "   ", "null"} {
		v, eErr := DecodeArgs(in)
		if eErr != nil {
			t.Fatalf("absent %q: %v", in, eErr)
		}
		if _, ok := v.(map[string]any); !ok {
			t.Fatalf("absent %q = %T, want empty object", in, v)
		}
	}
	v, eErr := DecodeArgs(`{"city":"sf","n":1}`)
	if eErr != nil {
		t.Fatalf("valid object: %v", eErr)
	}
	m, ok := v.(map[string]any)
	if !ok || m["city"] != "sf" {
		t.Fatalf("parsed = %#v", v)
	}
	_, eErr = DecodeArgs("{oops")
	if eErr == nil || eErr.Class != errclass.ClassTranslation ||
		!strings.Contains(eErr.Message, "malformed tool call arguments JSON") {
		t.Fatalf("malformed = %v", eErr)
	}
}

func TestImageSource(t *testing.T) {
	src, eErr := ImageSource("https://cdn.test/pic.png")
	if eErr != nil {
		t.Fatalf("https url: %v", eErr)
	}
	want := map[string]any{
		"type":   "image",
		"source": map[string]any{"type": "url", "url": "https://cdn.test/pic.png"},
	}
	if !equalJSON(src, want) {
		t.Fatalf("url source = %#v", src)
	}

	base64Cases := []struct {
		url   string
		media string
		data  string
	}{
		{"data:image/png;base64,AAA", "image/png", "AAA"},
		{"data:;base64,AAA", "application/octet-stream", "AAA"},
	}
	for _, c := range base64Cases {
		src, eErr := ImageSource(c.url)
		if eErr != nil {
			t.Fatalf("%s: %v", c.url, eErr)
		}
		source := src["source"].(map[string]any)
		if source["type"] != "base64" || source["media_type"] != c.media || source["data"] != c.data {
			t.Fatalf("%s decoded to %#v", c.url, src)
		}
		if src["type"] != "image" {
			t.Fatalf("%s outer type = %v", c.url, src["type"])
		}
	}

	passthrough := []string{
		"data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3C%2Fsvg%3E",
		"data:text/plain,hi",
	}
	malformed := []string{
		"data:no-comma-here",
	}
	for _, u := range passthrough {
		src, eErr := ImageSource(u)
		if eErr != nil {
			t.Fatalf("%s: %v", u, eErr)
		}
		want := map[string]any{
			"type":   "image",
			"source": map[string]any{"type": "url", "url": u},
		}
		if !equalJSON(src, want) {
			t.Fatalf("%s decoded to %#v, want verbatim url source", u, src)
		}
	}
	for _, u := range malformed {
		if _, eErr := ImageSource(u); eErr == nil || eErr.Class != errclass.ClassTranslation {
			t.Fatalf("%s: want ClassTranslation, got %+v", u, eErr)
		}
	}
}

func TestSSEFraming(t *testing.T) {
	payload := json.RawMessage(`{"a":1}`)
	if got := string(SSEEvent("n", payload)); got != "event: n\ndata: {\"a\":1}\n\n" {
		t.Fatalf("SSEEvent = %q", got)
	}
}

// ChatChunkBuilder frame shapes are pinned byte-exact; Go marshals maps
// with sorted keys, so the rendered order is stable.
func TestChatChunkBuilderFrameShapes(t *testing.T) {
	b := ChatChunkBuilder{ID: "msg_1", Model: "minimax", Created: 1700000000}
	envelope := `"created":1700000000,"id":"msg_1","model":"minimax","object":"chat.completion.chunk"`

	got, want := string(b.RoleChunk()),
		`{"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":null,"index":0}],`+envelope+"}"
	if got != want {
		t.Fatalf("RoleChunk = %q, want %q", got, want)
	}

	got, want = string(b.Delta(map[string]any{"content": "hi"})),
		`{"choices":[{"delta":{"content":"hi"},"finish_reason":null,"index":0}],`+envelope+"}"
	if got != want {
		t.Fatalf("Delta = %q, want %q", got, want)
	}

	got, want = string(b.Finish("tool_calls", CCUsageFrom(10, 7, UsageDetails{}))),
		`{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}],`+envelope+
			`,"usage":{"completion_tokens":7,"prompt_tokens":10,"total_tokens":17}}`
	if got != want {
		t.Fatalf("Finish = %q, want %q", got, want)
	}

	got, want = string(b.Finish("", nil)),
		`{"choices":[{"delta":{},"finish_reason":null,"index":0}],`+envelope+"}"
	if got != want {
		t.Fatalf("Finish(empty) = %q, want %q", got, want)
	}
}

// A zero Created defaults to render-time now.
func TestChatChunkBuilderDefaults(t *testing.T) {
	b := ChatChunkBuilder{ID: "i", Model: "m"}
	var f struct {
		Created int64 `json:"created"`
	}
	if err := json.Unmarshal(b.RoleChunk(), &f); err != nil {
		t.Fatal(err)
	}
	if f.Created <= 0 {
		t.Errorf("zero created not defaulted to now: %d", f.Created)
	}
}

// ClaudeEventEmitter frames are pinned byte-exact against the canonical
// Messages shapes (Go marshals maps with sorted keys): exact event names,
// output_tokens:0 in message_start usage, and no stop_sequence member in
// message_delta.
func TestClaudeEventEmitterFrameShapes(t *testing.T) {
	e := ClaudeEventEmitter{id: "msg_1", model: "minimax"}

	got, want := string(e.MessageStart(7)),
		"event: message_start\n"+
			`data: {"message":{"content":[],"id":"msg_1","model":"minimax","role":"assistant","type":"message","usage":{"input_tokens":7,"output_tokens":0}},"type":"message_start"}`+"\n\n"
	if got != want {
		t.Fatalf("MessageStart = %q, want %q", got, want)
	}

	got, want = string(e.ContentBlockStart(0, "text", map[string]any{"text": ""})),
		"event: content_block_start\n"+
			`data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}`+"\n\n"
	if got != want {
		t.Fatalf("ContentBlockStart(text) = %q, want %q", got, want)
	}

	got, want = string(e.ContentBlockStart(1, "tool_use",
		map[string]any{"id": "t1", "name": "f", "input": map[string]any{}})),
		"event: content_block_start\n"+
			`data: {"content_block":{"id":"t1","input":{},"name":"f","type":"tool_use"},"index":1,"type":"content_block_start"}`+"\n\n"
	if got != want {
		t.Fatalf("ContentBlockStart(tool_use) = %q, want %q", got, want)
	}

	got, want = string(e.ContentBlockDelta(1, map[string]any{"type": "text_delta", "text": "hi"})),
		"event: content_block_delta\n"+
			`data: {"delta":{"text":"hi","type":"text_delta"},"index":1,"type":"content_block_delta"}`+"\n\n"
	if got != want {
		t.Fatalf("ContentBlockDelta = %q, want %q", got, want)
	}

	got, want = string(e.ContentBlockStop(2)),
		"event: content_block_stop\n"+
			`data: {"index":2,"type":"content_block_stop"}`+"\n\n"
	if got != want {
		t.Fatalf("ContentBlockStop = %q, want %q", got, want)
	}

	stop := "end_turn"
	got, want = string(e.MessageDelta(&stop, map[string]any{"input_tokens": 4, "output_tokens": 9})),
		"event: message_delta\n"+
			`data: {"delta":{"stop_reason":"end_turn"},"type":"message_delta","usage":{"input_tokens":4,"output_tokens":9}}`+"\n\n"
	if got != want {
		t.Fatalf("MessageDelta = %q, want %q", got, want)
	}
	if strings.Contains(got, "stop_sequence") {
		t.Fatalf("message_delta must omit stop_sequence entirely: %q", got)
	}

	got = string(e.MessageDelta(nil, nil))
	want = "event: message_delta\n" + `data: {"delta":{},"type":"message_delta"}` + "\n\n"
	if got != want {
		t.Fatalf("MessageDelta(nil) = %q, want %q", got, want)
	}

	got, want = string(e.MessageStop()),
		"event: message_stop\n"+`data: {"type":"message_stop"}`+"\n\n"
	if got != want {
		t.Fatalf("MessageStop = %q, want %q", got, want)
	}
}

func TestRedactedSnippet(t *testing.T) {
	if got := RedactedSnippet("Bearer sk-secret-123 rest"); got != "Bearer [redacted] rest" {
		t.Fatalf("redaction = %q", got)
	}
	long := strings.Repeat("a", 100)
	got := RedactedSnippet(long)
	if got != long[:80]+"..." {
		t.Fatalf("truncation = %d chars, tail %q", len(got), got[len(got)-4:])
	}
	short := "plain text"
	if got := RedactedSnippet(short); got != short {
		t.Fatalf("short string changed: %q", got)
	}
}

func TestUpstreamStatusError(t *testing.T) {
	e := UpstreamStatusError(429, []byte("Bearer sk-secret-123 quota exhausted"))
	if e.Class != errclass.ClassQuota || e.StatusCode != 429 {
		t.Fatalf("classification = %+v", e)
	}
	if strings.Contains(e.Message, "sk-secret-123") || !strings.Contains(e.Message, "[redacted]") {
		t.Fatalf("secret leaked: %q", e.Message)
	}

	// Oversized bodies are truncated before snippet extraction, so the
	// message stays snippet-sized regardless of body size.
	for _, size := range []int{100, 4096, 4097, 1 << 20} {
		e = UpstreamStatusError(500, []byte(strings.Repeat("x", size)))
		if len(e.Message) > 83 || !strings.HasSuffix(e.Message, "...") {
			t.Fatalf("size %d not bounded: %d chars", size, len(e.Message))
		}
	}

	// Short bodies pass through as the whole redacted snippet.
	if e := UpstreamStatusError(503, []byte("down")); e.Message != "down" {
		t.Fatalf("short body = %q", e.Message)
	}
}

func equalJSON(a, b any) bool {
	ab, _ := json.Marshal(a)
	bb, _ := json.Marshal(b)
	return string(ab) == string(bb)
}

func TestOpenAIImageURL(t *testing.T) {
	url, eErr := OpenAIImageURL(json.RawMessage(`{"url":"https://x/a.png"}`))
	if eErr != nil || url != "https://x/a.png" {
		t.Fatalf("object form = %q %v", url, eErr)
	}
	url, eErr = OpenAIImageURL(json.RawMessage(`"https://x/b.jpg"`))
	if eErr != nil || url != "https://x/b.jpg" {
		t.Fatalf("string form = %q %v", url, eErr)
	}
	for name, raw := range map[string]json.RawMessage{
		"absent":       nil,
		"empty object": json.RawMessage(`{}`),
		"empty string": json.RawMessage(`""`),
	} {
		if _, eErr := OpenAIImageURL(raw); eErr == nil ||
			!strings.Contains(eErr.Message, "image content part missing url") {
			t.Fatalf("%s: err = %v", name, eErr)
		}
	}
}

func TestFunctionTool(t *testing.T) {
	if eErr := FunctionTool("function", "/v1/responses"); eErr != nil {
		t.Fatalf("valid tool: %v", eErr)
	}

	eErr := FunctionTool("web_search", "/v1/chat/completions")
	if eErr == nil || eErr.Class != errclass.ClassUnsupported ||
		!strings.Contains(eErr.Message, `"web_search"`) ||
		!strings.Contains(eErr.Message, "/v1/chat/completions") {
		t.Fatalf("rejection = %v", eErr)
	}
}

func TestUnsupportedFormat(t *testing.T) {
	eErr := UnsupportedFormat("grpc", "/v1/responses")
	if eErr == nil || eErr.Class != errclass.ClassUnsupported ||
		eErr.Message != `source protocol "grpc" cannot be translated to /v1/responses` {
		t.Fatalf("err = %+v", eErr)
	}
}

func TestValidationKernel(t *testing.T) {
	eErr := ValidateContentPart("audio", "/v1/messages")
	if eErr == nil || eErr.Class != errclass.ClassUnsupported ||
		eErr.Message != `unsupported content part type "audio" for /v1/messages` {
		t.Fatalf("part = %+v", eErr)
	}
	eErr = ValidateRole("robot", "/v1/chat/completions")
	if eErr == nil || eErr.Class != errclass.ClassUnsupported ||
		eErr.Message != `unsupported message role "robot" for /v1/chat/completions` {
		t.Fatalf("role = %+v", eErr)
	}
}

func TestIsContentPartType(t *testing.T) {
	for _, accepted := range []string{"text", "input_text", "output_text", "image_url", "input_image"} {
		if !IsContentPartType(accepted) {
			t.Errorf("IsContentPartType(%q) = false, want true", accepted)
		}
	}
	for _, rejected := range []string{"audio", "video", ""} {
		if IsContentPartType(rejected) {
			t.Errorf("IsContentPartType(%q) = true, want false", rejected)
		}
	}
}

func TestClaudeSystemText(t *testing.T) {
	if got, _ := ClaudeSystemText(nil); got != "" {
		t.Errorf("absent system = %q", got)
	}
	if got, _ := ClaudeSystemText(json.RawMessage(`"plain"`)); got != "plain" {
		t.Errorf("string system = %q", got)
	}
	got, eErr := ClaudeSystemText(json.RawMessage(`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`))
	if eErr != nil || got != "a\n\nb" {
		t.Errorf("text blocks = %q, %v; want a\\n\\nb, nil", got, eErr)
	}
	if _, eErr := ClaudeSystemText(json.RawMessage(`[{"type":"doc","text":"x"}]`)); eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Errorf("non-text block err = %+v", eErr)
	}
	if _, eErr := ClaudeSystemText(json.RawMessage(`42`)); eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Errorf("malformed system err = %+v", eErr)
	}
}

func TestToolResultText(t *testing.T) {
	for _, raw := range []json.RawMessage{nil, json.RawMessage(`null`)} {
		if got, _ := ToolResultText(raw, "n"); got != "" {
			t.Errorf("absent content = %q", got)
		}
	}
	if got, _ := ToolResultText(json.RawMessage(`"plain"`), "n"); got != "plain" {
		t.Errorf("string content = %q", got)
	}
	got, eErr := ToolResultText(json.RawMessage(`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`), "n")
	if eErr != nil || got != "ab" {
		t.Errorf("block array = %q, %v; want ab, nil", got, eErr)
	}
	if _, eErr := ToolResultText(json.RawMessage(`[{"type":"image","source":{}}]`), "tool messages carry text only"); eErr == nil ||
		eErr.Class != errclass.ClassTranslation ||
		eErr.Message != `unsupported tool_result block type "image"; tool messages carry text only` {
		t.Errorf("non-text block err = %+v", eErr)
	}
	if _, eErr := ToolResultText(json.RawMessage(`42`), "n"); eErr == nil || eErr.Class != errclass.ClassTranslation {
		t.Errorf("malformed content err = %+v", eErr)
	}
}

func TestStopFinishRoundTrip(t *testing.T) {
	for _, stop := range []string{"tool_use", "max_tokens", "refusal"} {
		if back := FinishToClaudeStop(ClaudeStopToFinish(stop)); back != stop {
			t.Fatalf("%s did not round-trip (got %q)", stop, back)
		}
	}
	for _, finish := range []string{"tool_calls", "length", "content_filter"} {
		if back := ClaudeStopToFinish(FinishToClaudeStop(finish)); back != finish {
			t.Fatalf("%s did not round-trip (got %q)", finish, back)
		}
	}
	if got := FinishToClaudeStop("semaphore"); got != "end_turn" {
		t.Fatalf("unknown finish = %q", got)
	}
}

func TestSSEFramer(t *testing.T) {
	t.Run("frames join multi data lines and skip strays and comments", func(t *testing.T) {
		f := NewSSEFramer(false)
		f.Push([]byte(": heartbeat\n\n")[:len(": heartbeat\n")]) // partial first push
		f.Push([]byte("\n"))
		f.Push([]byte("event: e1\ndata: one\n"))
		f.Push([]byte("data: two\n\r\n")) // CRLF blank terminates
		f.Push([]byte("\n"))              // stray blank after frame
		var frames []string
		for {
			etype, data, raw, ok := f.Next()
			if !ok {
				break
			}
			if raw != nil {
				t.Fatal("raw captured with wantRaw=false")
			}
			frames = append(frames, etype+"|"+data)
		}
		if len(frames) != 1 || frames[0] != "e1|one\ntwo" {
			t.Fatalf("frames = %q", frames)
		}
	})

	t.Run("raw capture only when wantRaw", func(t *testing.T) {
		f := NewSSEFramer(true)
		f.Push([]byte("event: x\ndata: 1\n\n"))
		_, data, raw, ok := f.Next()
		if !ok || data != "1" || string(raw) != "event: x\ndata: 1\n\n" {
			t.Fatalf("frame = %q %q", data, raw)
		}
		// State reset: next frame starts clean.
		f.Push([]byte("data: 2\n\n"))
		_, data, raw, ok = f.Next()
		if !ok || data != "2" || string(raw) != "data: 2\n\n" {
			t.Fatalf("second frame = %q %q", data, raw)
		}
	})
}

func TestObjectSchema(t *testing.T) {
	if got := string(ObjectSchema(nil)); got != `{"type":"object","properties":{}}` {
		t.Fatalf("absent = %s", got)
	}
	if got := string(ObjectSchema(json.RawMessage("null"))); got != `{"type":"object","properties":{}}` {
		t.Fatalf("null = %s", got)
	}
	given := json.RawMessage(`{"type":"string"}`)
	if got := string(ObjectSchema(given)); got != `{"type":"string"}` {
		t.Fatalf("present schema altered: %s", got)
	}
	typed := json.RawMessage(`{"type":"object","properties":{}}`)
	if got := string(ObjectSchema(typed)); got != string(typed) {
		t.Fatalf("typed object altered: %s", got)
	}
	injected := ObjectSchema(json.RawMessage(`{"anyOf":[{"type":"object","properties":{"agent":{"type":"string"}}}]}`))
	var obj map[string]any
	if err := json.Unmarshal(injected, &obj); err != nil {
		t.Fatal(err)
	}
	if obj["type"] != "object" {
		t.Fatalf("anyOf without type: %s", injected)
	}
	if _, ok := obj["anyOf"]; !ok {
		t.Fatalf("anyOf dropped: %s", injected)
	}
	propsOnly := ObjectSchema(json.RawMessage(`{"properties":{}}`))
	if err := json.Unmarshal(propsOnly, &obj); err != nil {
		t.Fatal(err)
	}
	if obj["type"] != "object" {
		t.Fatalf("properties without type: %s", propsOnly)
	}
}

func TestClaudeStopToFinish(t *testing.T) {
	cases := map[string]string{
		"end_turn":      "stop",
		"stop_sequence": "stop",
		"tool_use":      "tool_calls",
		"max_tokens":    "length",
		"refusal":       "content_filter",
		"something_new": "stop",
	}
	for stop, want := range cases {
		if got := ClaudeStopToFinish(stop); got != want {
			t.Errorf("ClaudeStopToFinish(%q) = %q, want %q", stop, got, want)
		}
	}
}

func TestTerminalReason(t *testing.T) {
	// Chat Completions vocabulary: tools outrank an incomplete status.
	if got := TerminalReason(true, "tool_calls", "length"); got != "tool_calls" {
		t.Errorf("tools+incomplete = %q", got)
	}
	// Messages vocabulary behaves identically.
	if got := TerminalReason(true, "tool_use", "max_tokens"); got != "tool_use" {
		t.Errorf("tools+incomplete (claude) = %q", got)
	}
	if got := TerminalReason(false, "tool_calls", "length"); got != "length" {
		t.Errorf("no tools = %q", got)
	}
	if got := TerminalReason(false, "tool_use", "end_turn"); got != "end_turn" {
		t.Errorf("no tools completed = %q", got)
	}
}

func TestNewClaudeResult(t *testing.T) {
	b, _ := json.Marshal(NewClaudeResult("r1", "m", "end_turn", 2, 3, nil, nil,
		[]map[string]any{{"type": "text", "text": "x"}}))
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("not json: %v (%s)", err, b)
	}
	if m["id"] != "r1" || m["type"] != "message" || m["role"] != "assistant" ||
		m["model"] != "m" || m["stop_reason"] != "end_turn" {
		t.Fatalf("envelope wrong: %v", m)
	}
	usage := m["usage"].(map[string]any)
	if usage["input_tokens"] != float64(2) || usage["output_tokens"] != float64(3) {
		t.Fatalf("usage wrong: %v", usage)
	}
	content := m["content"].([]any)
	if len(content) != 1 || content[0].(map[string]any)["text"] != "x" {
		t.Fatalf("content wrong: %v", content)
	}

	// Canonical omissions: nil content renders as [], and stop_sequence is
	// never present (the stream message_delta sibling omits it entirely).
	b, _ = json.Marshal(NewClaudeResult("r2", "m", "max_tokens", 0, 0, nil, nil, nil))
	s := string(b)
	if strings.Contains(s, `"content":null`) || strings.Contains(s, "stop_sequence") {
		t.Fatalf("canonical omissions violated: %s", s)
	}
}

func TestClaudeImageURL(t *testing.T) {
	url, eErr := ClaudeImageURL("url", "https://x/i.png", "", "")
	if eErr != nil || url != "https://x/i.png" {
		t.Fatalf("url source = %q %v", url, eErr)
	}
	url, eErr = ClaudeImageURL("base64", "", "image/png", "QUJD")
	if eErr != nil || url != "data:image/png;base64,QUJD" {
		t.Fatalf("base64 source = %q %v", url, eErr)
	}
	url, eErr = ClaudeImageURL("base64", "", "", "QUJD")
	if eErr != nil || url != "data:application/octet-stream;base64,QUJD" {
		t.Fatalf("default media type = %q %v", url, eErr)
	}
	for name, tc := range map[string]struct {
		srcType, want string
	}{
		"url empty":    {"url", "image source missing url"},
		"data empty":   {"base64", "base64 image source missing data"},
		"unknown type": {"s3", `unsupported image source type "s3"`},
	} {
		if _, eErr := ClaudeImageURL(tc.srcType, "", "", ""); eErr == nil ||
			eErr.Class != errclass.ClassTranslation || eErr.Message != tc.want {
			t.Fatalf("%s: err = %v", name, eErr)
		}
	}
}

func TestRewriteModelIDPreservesBigIntegers(t *testing.T) {
	body := []byte(`{"model":"x","seed":9007199254740993,"nested":{"ts":1700000000123}}`)
	out, eErr := RewriteModelID("glm-5.2", body, "openai")
	if eErr != nil {
		t.Fatalf("rewrite: %v", eErr)
	}
	// A float64 roundtrip would corrupt 9007199254740993 (> 2^53); raw
	// passthrough must preserve it byte-exact.
	if !strings.Contains(string(out), "9007199254740993") ||
		!strings.Contains(string(out), "1700000000123") {
		t.Fatalf("big integers mangled: %s", out)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil || got["model"] != "glm-5.2" {
		t.Fatalf("model rewrite wrong: %v %s", err, out)
	}
}

// A catalog model id containing a quote cannot form a valid JSON string;
// the rewrite must fail with a Translation error, never nil-body-nil-error
// (which would POST an empty upstream body).
func TestRewriteModelIDUnrepresentableModelID(t *testing.T) {
	out, eErr := RewriteModelID(`we"ird`, []byte(`{"model":"x","messages":[]}`), "openai")
	if eErr == nil || eErr.Class != errclass.ClassTranslation ||
		!strings.Contains(eErr.Message, "model id cannot be represented as JSON") {
		t.Fatalf("err = %v, out = %s", eErr, out)
	}
	if out != nil {
		t.Fatalf("body must be nil on failure, got %s", out)
	}
}

// FR-005 drift fix: positive max_tokens passes through; absent/zero/
// negative defaults to 4096 so every Claude-source leg applies one policy.
func TestClaudeMaxTokens(t *testing.T) {
	for _, tc := range []struct{ in, want int64 }{
		{1, 1}, {256, 256}, {0, 4096}, {-5, 4096},
	} {
		if got := ClaudeMaxTokens(tc.in); got != tc.want {
			t.Fatalf("ClaudeMaxTokens(%d) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

func TestDecodeToolChoice(t *testing.T) {
	if k, n, eErr := DecodeToolChoice(nil); k != ToolChoiceAbsent || eErr != nil {
		t.Fatalf("absent = %q %q %v", k, n, eErr)
	}
	if k, _, eErr := DecodeToolChoice(json.RawMessage(`"auto"`)); k != ToolChoiceAuto || eErr != nil {
		t.Fatalf("string auto = %q %v", k, eErr)
	}
	if k, _, eErr := DecodeToolChoice(json.RawMessage(`"required"`)); k != ToolChoiceAny || eErr != nil {
		t.Fatalf("string required = %q %v", k, eErr)
	}
	if k, _, eErr := DecodeToolChoice(json.RawMessage(`"none"`)); k != ToolChoiceNone || eErr != nil {
		t.Fatalf("string none = %q %v", k, eErr)
	}
	if k, _, eErr := DecodeToolChoice(json.RawMessage(`{"type":"auto"}`)); k != ToolChoiceAuto || eErr != nil {
		t.Fatalf("object auto = %q %v", k, eErr)
	}
	if k, _, eErr := DecodeToolChoice(json.RawMessage(`{"type":"any"}`)); k != ToolChoiceAny || eErr != nil {
		t.Fatalf("object any = %q %v", k, eErr)
	}
	if k, n, eErr := DecodeToolChoice(json.RawMessage(`{"type":"required","name":"z"}`)); k != ToolChoiceAny || eErr != nil {
		t.Fatalf("required named = %q %q %v", k, n, eErr)
	}
	if k, n, eErr := DecodeToolChoice(json.RawMessage(`{"type":"function","name":"h"}`)); k != ToolChoiceNamed || n != "h" || eErr != nil {
		t.Fatalf("flat named = %q %q %v", k, n, eErr)
	}
	if k, _, _ := DecodeToolChoice(json.RawMessage(`{"type":"function","function":{"name":"f"},"name":"ignored"}`)); k != ToolChoiceNamed {
		t.Fatalf("nested-name precedence = %q", k)
	}
	if k, _, eErr := DecodeToolChoice(json.RawMessage(`{"type":"none"}`)); k != ToolChoiceNone || eErr != nil {
		t.Fatalf("none = %q %v", k, eErr)
	}
	k, name, eErr := DecodeToolChoice(json.RawMessage(`{"type":"function","function":{"name":"f"}}`))
	if k != ToolChoiceNamed || name != "f" || eErr != nil {
		t.Fatalf("CC named = %q %q %v", k, name, eErr)
	}
	k, name, eErr = DecodeToolChoice(json.RawMessage(`{"type":"tool","name":"g"}`))
	if k != ToolChoiceNamed || name != "g" || eErr != nil {
		t.Fatalf("claude named = %q %q %v", k, name, eErr)
	}
	_, _, eErr = DecodeToolChoice(json.RawMessage(`{"type":"tool"}`))
	if eErr == nil || !strings.Contains(eErr.Message, "named tool requires a name") {
		t.Fatalf("nameless named = %v", eErr)
	}
	_, _, eErr = DecodeToolChoice(json.RawMessage(`{bad`))
	if eErr == nil || !strings.Contains(eErr.Message, "malformed tool_choice") {
		t.Fatalf("malformed = %v", eErr)
	}
	_, _, eErr = DecodeToolChoice(json.RawMessage(`{"type":"web_search"}`))
	if eErr == nil || eErr.Class != errclass.ClassTranslation ||
		!strings.Contains(eErr.Message, `"web_search"`) {
		t.Fatalf("unsupported type = %v", eErr)
	}
}

// Unknown SSE lines (retry:, id:, anything unrecognized) are ignored
// outside raw capture.
func TestSSEFramerIgnoresUnknownLines(t *testing.T) {
	f := NewSSEFramer(true)
	f.Push([]byte("retry: 500\nid: zzz\ndata: {\"a\":1}\n\n"))
	etype, data, raw, ok := f.Next()
	if !ok || etype != "" || data != `{"a":1}` {
		t.Fatalf("frame = %q %q", etype, data)
	}
	if string(raw) != "retry: 500\nid: zzz\ndata: {\"a\":1}\n\n" {
		t.Fatalf("raw = %q", raw)
	}
}

// Full finish_reason → status table; unknown and empty inputs default to
// completed exactly as the inline maps did.
func TestResponseStatusFromCCFinish(t *testing.T) {
	for finish, want := range map[string]string{
		"stop":           "completed",
		"length":         "incomplete",
		"tool_calls":     "completed",
		"function_call":  "completed",
		"content_filter": "completed",
		"":               "completed",
		"nonsense":       "completed",
	} {
		if got := ResponseStatusFromCCFinish(finish); got != want {
			t.Errorf("ResponseStatusFromCCFinish(%q) = %q, want %q", finish, got, want)
		}
	}
}

// Total is always input+output regardless of inputs; zero stays zero.
func TestNewResponsesUsageFrom(t *testing.T) {
	for _, tc := range []struct {
		in, out, wantIn, wantOut, wantTotal int64
	}{
		{0, 0, 0, 0, 0},
		{5, 7, 5, 7, 12},
	} {
		got := NewResponsesUsageFrom(tc.in, tc.out, UsageDetails{})
		if got.InputTokens != tc.wantIn || got.OutputTokens != tc.wantOut || got.TotalTokens != tc.wantTotal {
			t.Errorf("NewResponsesUsageFrom(%d,%d) = %+v, want in=%d out=%d total=%d",
				tc.in, tc.out, got, tc.wantIn, tc.wantOut, tc.wantTotal)
		}
	}
}

// Full stop_reason → status table; unknown and empty inputs default to
// completed exactly as the inline maps did.
func TestResponseStatusFromClaudeStop(t *testing.T) {
	for stop, want := range map[string]string{
		"end_turn":      "completed",
		"max_tokens":    "incomplete",
		"stop_sequence": "completed",
		"tool_use":      "completed",
		"":              "completed",
		"nonsense":      "completed",
	} {
		if got := ResponseStatusFromClaudeStop(stop); got != want {
			t.Errorf("ResponseStatusFromClaudeStop(%q) = %q, want %q", stop, got, want)
		}
	}
}

// SystemImageRejected carries the canonical ClassTranslation wording every
// adapter shares; the message text is pinned byte-exact.
func TestSystemImageRejected(t *testing.T) {
	e := SystemImageRejected()
	if e == nil || e.Class != errclass.ClassTranslation ||
		e.Message != "system messages cannot carry image content" {
		t.Fatalf("SystemImageRejected() = %+v", e)
	}
}

// Usage maps carry the protocol vocabulary with computed totals
// (majority rule), including zero-token results.
func TestUsageFrom(t *testing.T) {
	got := CCUsageFrom(10, 3, UsageDetails{})
	want := map[string]any{"prompt_tokens": int64(10), "completion_tokens": int64(3), "total_tokens": int64(13)}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("CCUsageFrom(10,3)[%s] = %v, want %v", k, got[k], v)
		}
	}
	if len(got) != 3 {
		t.Errorf("CCUsageFrom(10,3) has %d keys, want 3", len(got))
	}
}

func TestUsageDetailsPreserveZeroAndClamp(t *testing.T) {
	zero := int64(0)
	read := int64(4)
	write := int64(3)
	reasoning := int64(0)
	cc := CCUsageFrom(7, 5, UsageDetails{CachedTokens: &zero, ReasoningTokens: &reasoning})
	if cc["prompt_tokens_details"].(map[string]any)["cached_tokens"] != int64(0) ||
		cc["completion_tokens_details"].(map[string]any)["reasoning_tokens"] != int64(0) {
		t.Fatalf("explicit zero details lost: %v", cc)
	}
	resp := NewResponsesUsageFrom(7, 5, UsageDetails{CachedTokens: &read, CacheWriteTokens: &write})
	if resp.InputDetails == nil || *resp.InputDetails.CachedTokens != 4 || *resp.InputDetails.CacheWriteTokens != 3 || resp.TotalTokens != 12 {
		t.Fatalf("responses details wrong: %+v", resp)
	}
	if got := ClampSubtract(2, &read, &write); got != 0 {
		t.Fatalf("ClampSubtract = %d, want 0", got)
	}
}

// Inverse status mappings and unsupported-input kernels: incomplete is
// the only truncating status; unknown/absent completes. The kernels must
// carry ClassUnsupported (FR-009), not translation.
func TestResponseStatusInverseKernels(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"incomplete", "length"}, {"completed", "stop"}, {"", "stop"}, {"nonsense", "stop"},
	} {
		if got := CCFinishFromResponseStatus(tc.in); got != tc.want {
			t.Errorf("CCFinishFromResponseStatus(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, tc := range []struct{ in, want string }{
		{"incomplete", "max_tokens"}, {"completed", "end_turn"}, {"", "end_turn"}, {"nonsense", "end_turn"},
	} {
		if got := ClaudeStopFromResponseStatus(tc.in); got != tc.want {
			t.Errorf("ClaudeStopFromResponseStatus(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}

	eErr := UnsupportedPartType("audio", "/v1/chat/completions")
	if eErr == nil || eErr.Class != errclass.ClassUnsupported ||
		eErr.Message != `unsupported content block type "audio" for /v1/chat/completions` {
		t.Errorf("UnsupportedPartType = %v", eErr)
	}
	eErr = UnsupportedInputItemType("web_search")
	if eErr == nil || eErr.Class != errclass.ClassUnsupported || eErr.StatusCode != 400 ||
		eErr.Message != `unsupported Responses input item type "web_search"` {
		t.Errorf("UnsupportedInputItemType = %v", eErr)
	}
}

// ThinkingEnabled gates on presence AND the "enabled" type: nil and any
// non-enabled control are false (FR-005).
func TestThinkingEnabled(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   *ClaudeThinking
		want bool
	}{
		{"nil control", nil, false},
		{"present but not enabled", &ClaudeThinking{Type: "disabled"}, false},
		{"enabled", &ClaudeThinking{Type: "enabled", BudgetTokens: 1024}, true},
	} {
		if got := ThinkingEnabled(tc.in); got != tc.want {
			t.Errorf("%s: ThinkingEnabled(%+v) = %v, want %v", tc.name, tc.in, got, tc.want)
		}
	}
}

// NewClaudeEventEmitter binds the stream identity the frames carry.
func TestNewClaudeEventEmitter(t *testing.T) {
	if got := NewClaudeEventEmitter("msg_9", "gpt"); got != (ClaudeEventEmitter{id: "msg_9", model: "gpt"}) {
		t.Fatalf("NewClaudeEventEmitter = %+v", got)
	}
}

// ssePayload decodes the data JSON of one rendered SSE event frame.
func ssePayload(t *testing.T, frame []byte) map[string]any {
	t.Helper()
	_, rest, ok := strings.Cut(string(frame), "\ndata: ")
	if !ok {
		t.Fatalf("frame has no data line: %q", frame)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSuffix(rest, "\n\n")), &m); err != nil {
		t.Fatalf("data payload: %v", err)
	}
	return m
}

// FR-006: the shared Responses emitter renders the canonical wire shapes —
// created_at falls back to render-time now, deltas carry item identity,
// and Completed always carries model and summed usage beside the output.
func TestResponsesEventEmitter(t *testing.T) {
	e := ResponsesEventEmitter{ID: "r1", Model: "m1"}

	created := ssePayload(t, e.Created())
	if created["type"] != "response.created" {
		t.Fatalf("created type = %v", created["type"])
	}
	resp := created["response"].(map[string]any)
	if resp["id"] != "r1" || resp["object"] != "response" || resp["status"] != "in_progress" {
		t.Fatalf("created response = %v", resp)
	}
	if ts, ok := resp["created_at"].(float64); !ok || ts <= 0 {
		t.Fatalf("created_at must fall back to render-time now, got %v", resp["created_at"])
	}

	added := ssePayload(t, e.ItemAdded(2, map[string]any{"type": "message", "id": "r1"}))
	if added["type"] != "response.output_item.added" || added["output_index"] != float64(2) ||
		added["item"].(map[string]any)["id"] != "r1" {
		t.Fatalf("item added = %v", added)
	}
	td := ssePayload(t, e.TextDelta("r1", 0, "hej"))
	if td["item_id"] != "r1" || td["output_index"] != float64(0) || td["delta"] != "hej" {
		t.Fatalf("text delta = %v", td)
	}
	ad := ssePayload(t, e.ArgsDelta("c1", 1, "{"))
	if ad["item_id"] != "c1" || ad["output_index"] != float64(1) || ad["delta"] != "{" {
		t.Fatalf("args delta = %v", ad)
	}

	done := ssePayload(t, e.Completed("completed", NewResponsesUsageFrom(2, 3, UsageDetails{}), []any{}))
	if done["type"] != "response.completed" {
		t.Fatalf("completed type = %v", done["type"])
	}
	resp = done["response"].(map[string]any)
	if resp["id"] != "r1" || resp["model"] != "m1" || resp["status"] != "completed" ||
		len(resp["output"].([]any)) != 0 {
		t.Fatalf("completed response = %v", resp)
	}
	u := resp["usage"].(map[string]any)
	if u["input_tokens"] != float64(2) || u["output_tokens"] != float64(3) || u["total_tokens"] != float64(5) {
		t.Fatalf("completed usage = %v", u)
	}
}

// msgItem is the message item Render synthesizes for the given text.
func msgItem(id, text string) RespItem {
	content, _ := json.Marshal([]outputTextPart{{Type: "output_text", Text: text}})
	return RespItem{Type: "message", ID: id, Role: "assistant", Content: content}
}

// OutputAssembler placement rules: a reserved slot pins the message position
// at reserve time and is filled even when the aggregated text is empty;
// unreserved sites lead with the message only when text was observed; an
// empty unreserved output stays empty.
func TestOutputAssembler(t *testing.T) {
	for _, tc := range []struct {
		name string
		run  func(a *OutputAssembler)
		want []any
	}{
		{
			name: "reserved slot filled empty-text, calls stay behind",
			run: func(a *OutputAssembler) {
				a.ReserveTextSlot()
				a.AppendFunctionCall("c1", "f", "{}")
			},
			want: []any{
				msgItem("m", ""),
				RespItem{Type: "function_call", CallID: "c1", Name: "f", Arguments: "{}"},
			},
		},
		{
			name: "reserve between calls interleaves in arrival order",
			run: func(a *OutputAssembler) {
				a.AppendFunctionCall("c1", "f", "")
				a.ReserveTextSlot()
				a.AppendFunctionCall("c2", "g", "")
				a.AddText("hi")
			},
			want: []any{
				RespItem{Type: "function_call", CallID: "c1", Name: "f"},
				msgItem("m", "hi"),
				RespItem{Type: "function_call", CallID: "c2", Name: "g"},
			},
		},
		{
			name: "reserve is idempotent",
			run: func(a *OutputAssembler) {
				a.ReserveTextSlot()
				a.AppendFunctionCall("c1", "f", "")
				a.ReserveTextSlot()
				a.AddText("t")
			},
			want: []any{
				msgItem("m", "t"),
				RespItem{Type: "function_call", CallID: "c1", Name: "f"},
			},
		},
		{
			name: "unreserved message leads when text observed",
			run: func(a *OutputAssembler) {
				a.AddText("t")
				a.AppendFunctionCall("c1", "f", "")
			},
			want: []any{
				msgItem("m", "t"),
				RespItem{Type: "function_call", CallID: "c1", Name: "f"},
			},
		},
		{
			name: "unreserved empty output stays empty",
			run:  func(a *OutputAssembler) {},
			want: []any{},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := NewOutputAssembler("m")
			tc.run(a)
			got := a.Render()
			if len(got) != len(tc.want) {
				t.Fatalf("Render() = %+v, want %d items", got, len(tc.want))
			}
			for i := range got {
				if !reflect.DeepEqual(got[i], tc.want[i]) {
					t.Errorf("Render()[%d] = %+v, want %+v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestCompletionEnvelope(t *testing.T) {
	choices := []map[string]any{{"index": 0}}
	usage := map[string]any{"total_tokens": 5}
	got := CompletionEnvelope("id-1", "glm-5.2", 1234, choices, usage)
	want := map[string]any{
		"id":      "id-1",
		"object":  "chat.completion",
		"created": int64(1234),
		"model":   "glm-5.2",
		"choices": choices,
		"usage":   usage,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("CompletionEnvelope() = %+v, want %+v", got, want)
	}
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(b, &keys); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, k := range []string{"id", "object", "created", "model", "choices", "usage"} {
		if _, ok := keys[k]; !ok {
			t.Errorf("wire shape missing %q", k)
		}
	}
	if len(keys) != len(want) {
		t.Errorf("wire shape has extra members: %v", keys)
	}
	var obj struct {
		Object string `json:"object"`
	}
	if err := json.Unmarshal(b, &obj); err != nil || obj.Object != "chat.completion" {
		t.Errorf("object = %q (%v), want chat.completion", obj.Object, err)
	}
}

// Opening-entry exact-shape pin: index/id/type/function{name,arguments}
// and nothing else — OpenAI streams put type on the first delta entry.
func TestCCToolCallOpeningEntry(t *testing.T) {
	got := CCToolCallOpeningEntry(2, "call_9", "lookup", `{"q"`)
	want := map[string]any{
		"index": 2,
		"id":    "call_9",
		"type":  "function",
		"function": map[string]any{
			"name":      "lookup",
			"arguments": `{"q"`,
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("CCToolCallOpeningEntry() = %+v, want %+v", got, want)
	}
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(b, &keys); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(keys) != 4 {
		t.Fatalf("wire shape member count = %d (%v), want 4", len(keys), keys)
	}
	for _, k := range []string{"index", "id", "type", "function"} {
		if _, ok := keys[k]; !ok {
			t.Errorf("wire shape missing %q", k)
		}
	}
}

// The kinds slice must carry exactly the five ToolChoice* constants in
// kind order — consumers' exhaustiveness tests iterate it.
func TestAllToolChoiceKinds(t *testing.T) {
	want := []string{
		ToolChoiceAbsent,
		ToolChoiceAuto,
		ToolChoiceAny,
		ToolChoiceNone,
		ToolChoiceNamed,
	}
	if !reflect.DeepEqual(AllToolChoiceKinds, want) {
		t.Fatalf("AllToolChoiceKinds = %v, want %v", AllToolChoiceKinds, want)
	}
	if len(AllToolChoiceKinds) != 5 {
		t.Fatalf("kinds = %d, want 5", len(AllToolChoiceKinds))
	}
}

func TestJoinTexts(t *testing.T) {
	for _, tc := range []struct {
		name  string
		parts []map[string]any
		want  string
	}{
		{"nil", nil, ""},
		{"empty slice", []map[string]any{}, ""},
		{"single", []map[string]any{{"text": "a"}}, "a"},
		{"join in order", []map[string]any{{"text": "a"}, {"text": "b"}}, "ab"},
		{"skip non-string", []map[string]any{{"text": "a"}, {"text": 3}, {"other": "x"}, {}, nil, {"text": "c"}}, "ac"},
	} {
		if got := JoinTexts(tc.parts); got != tc.want {
			t.Errorf("%s: JoinTexts = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// DecodeStringOrParts owns the empty-input policy for every target route:
// absent/null, empty strings, and empty-text parts all decode to zero
// parts; known part types keep arrival order; images resolve through
// OpenAIImageURL; unknown types are ClassUnsupported naming the target.
func TestDecodeStringOrParts(t *testing.T) {
	for name, raw := range map[string]json.RawMessage{
		"absent": nil,
		"null":   json.RawMessage(`null`),
	} {
		parts, eErr := DecodeStringOrParts(raw, "/v1/messages")
		if eErr != nil || parts != nil {
			t.Errorf("%s: = %+v, %v; want nil, nil", name, parts, eErr)
		}
	}

	if parts, eErr := DecodeStringOrParts(json.RawMessage(`""`), "/v1/messages"); eErr != nil || parts != nil {
		t.Errorf("empty string = %+v, %v; want nil, nil", parts, eErr)
	}
	if parts, eErr := DecodeStringOrParts(json.RawMessage(`"hi"`), "/v1/messages"); eErr != nil ||
		len(parts) != 1 || parts[0] != (ContentPart{Text: "hi"}) {
		t.Errorf("plain string = %+v, %v", parts, eErr)
	}

	if _, eErr := DecodeStringOrParts(json.RawMessage(`{oops`), "/v1/messages"); eErr == nil ||
		eErr.Class != errclass.ClassTranslation || !strings.Contains(eErr.Message, "malformed message content") {
		t.Errorf("malformed = %v", eErr)
	}

	// Ordered mixed parts: every accepted alias decodes in arrival order,
	// images in both wire shapes.
	raw := json.RawMessage(`[
		{"type":"input_text","text":"a"},
		{"type":"image_url","image_url":{"url":"https://x/i.png"}},
		{"type":"input_image","image_url":"https://y/j.jpg"},
		{"type":"output_text","text":"b"}
	]`)
	want := []ContentPart{
		{Text: "a"},
		{ImageURL: "https://x/i.png"},
		{ImageURL: "https://y/j.jpg"},
		{Text: "b"},
	}
	got, eErr := DecodeStringOrParts(raw, "/v1/responses")
	if eErr != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("ordered parts = %+v, %v; want %+v", got, eErr, want)
	}

	// Uniform DROP policy: empty-text parts ship nothing upstream.
	got, eErr = DecodeStringOrParts(json.RawMessage(
		`[{"type":"text","text":""},{"type":"text","text":"z"},{"type":"text","text":""}]`), "/v1/messages")
	if eErr != nil || len(got) != 1 || got[0].Text != "z" {
		t.Fatalf("empty-text drop = %+v, %v; want [z]", got, eErr)
	}
	got, eErr = DecodeStringOrParts(json.RawMessage(`[{"type":"text","text":""}]`), "/v1/messages")
	if eErr != nil || got != nil {
		t.Fatalf("all-empty parts = %+v, %v; want nil, nil", got, eErr)
	}

	if _, eErr := DecodeStringOrParts(json.RawMessage(
		`[{"type":"image_url","image_url":{}}]`), "/v1/messages"); eErr == nil ||
		eErr.Class != errclass.ClassTranslation || !strings.Contains(eErr.Message, "missing url") {
		t.Errorf("url-less image = %v", eErr)
	}

	if _, eErr := DecodeStringOrParts(json.RawMessage(
		`[{"type":"audio"}]`), "/v1/chat/completions"); eErr == nil ||
		eErr.Class != errclass.ClassUnsupported ||
		eErr.Message != `unsupported content part type "audio" for /v1/chat/completions` {
		t.Errorf("unknown part = %v", eErr)
	}
}

// ClaudeBlocksFromOpenAI owns the openai-content→Anthropic-blocks kernel:
// empties drop, strings and parts map to text/image blocks in arrival
// order, unknown part types are ClassUnsupported naming the target.
func TestClaudeBlocksFromOpenAI(t *testing.T) {
	for name, raw := range map[string]json.RawMessage{
		"absent": nil,
		"null":   json.RawMessage(`null`),
	} {
		blocks, eErr := ClaudeBlocksFromOpenAI(raw, "/v1/messages")
		if eErr != nil || blocks != nil {
			t.Errorf("%s: = %+v, %v; want nil, nil", name, blocks, eErr)
		}
	}

	blocks, eErr := ClaudeBlocksFromOpenAI(json.RawMessage(`"hi"`), "/v1/messages")
	if eErr != nil || len(blocks) != 1 ||
		!reflect.DeepEqual(blocks[0], map[string]any{"type": "text", "text": "hi"}) {
		t.Fatalf("plain string = %+v, %v", blocks, eErr)
	}

	blocks, eErr = ClaudeBlocksFromOpenAI(json.RawMessage(
		`[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}},{"type":"input_text","text":""}]`),
		"/v1/messages")
	if eErr != nil || len(blocks) != 2 {
		t.Fatalf("mixed parts = %+v, %v", blocks, eErr)
	}
	if blocks[0]["type"] != "text" || blocks[0]["text"] != "a" {
		t.Errorf("text block = %v", blocks[0])
	}
	src, ok := blocks[1]["source"].(map[string]any)
	if !ok || blocks[1]["type"] != "image" || src["type"] != "base64" || src["media_type"] != "image/png" {
		t.Errorf("image block = %v", blocks[1])
	}

	if _, eErr := ClaudeBlocksFromOpenAI(json.RawMessage(`{oops`), "/v1/messages"); eErr == nil ||
		eErr.Class != errclass.ClassTranslation {
		t.Errorf("malformed = %v", eErr)
	}
	if _, eErr := ClaudeBlocksFromOpenAI(json.RawMessage(`[{"type":"audio"}]`), "/v1/chat/completions"); eErr == nil ||
		eErr.Class != errclass.ClassUnsupported ||
		eErr.Message != `unsupported content part type "audio" for /v1/chat/completions` {
		t.Errorf("unknown part = %v", eErr)
	}
}

// DecodeInstructions and DecodeInputItems own the Responses envelope
// decode shared by both Responses-source builders: absent/null decode to
// zero values, plain-string input surfaces as one synthesized user
// message item, other shapes fail with the canonical wording.
func TestResponsesRequestDecodeHelpers(t *testing.T) {
	var absent ResponsesRequest
	if s, eErr := absent.DecodeInstructions(); eErr != nil || s != "" {
		t.Errorf("absent instructions = %q, %v", s, eErr)
	}
	if items, eErr := absent.DecodeInputItems(); eErr != nil || items != nil {
		t.Errorf("absent input = %+v, %v", items, eErr)
	}

	var req ResponsesRequest
	if err := json.Unmarshal([]byte(`{"instructions":"be nice","input":"hello"}`), &req); err != nil {
		t.Fatal(err)
	}
	if s, eErr := req.DecodeInstructions(); eErr != nil || s != "be nice" {
		t.Errorf("instructions = %q, %v", s, eErr)
	}
	items, eErr := req.DecodeInputItems()
	if eErr != nil || len(items) != 1 {
		t.Fatalf("string input = %+v, %v", items, eErr)
	}
	var text string
	if it := items[0]; it.Type != "message" || it.Role != "user" ||
		json.Unmarshal(it.Content, &text) != nil || text != "hello" {
		t.Errorf("synthesized item = %+v", items[0])
	}

	var emptyIn ResponsesRequest
	if err := json.Unmarshal([]byte(`{"input":""}`), &emptyIn); err != nil {
		t.Fatal(err)
	}
	items, eErr = emptyIn.DecodeInputItems()
	if eErr != nil || len(items) != 1 || string(items[0].Content) != `""` {
		t.Errorf("empty-string input = %+v, %v", items, eErr)
	}

	var nullIn ResponsesRequest
	if err := json.Unmarshal([]byte(`{"input":null}`), &nullIn); err != nil {
		t.Fatal(err)
	}
	if items, eErr := nullIn.DecodeInputItems(); eErr != nil || items != nil {
		t.Errorf("null input = %+v, %v", items, eErr)
	}

	var arrIn ResponsesRequest
	if err := json.Unmarshal([]byte(
		`{"input":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]}`), &arrIn); err != nil {
		t.Fatal(err)
	}
	items, eErr = arrIn.DecodeInputItems()
	if eErr != nil || len(items) != 1 || items[0].Type != "function_call" || items[0].CallID != "c" {
		t.Errorf("array input = %+v, %v", items, eErr)
	}

	var badInstr ResponsesRequest
	if err := json.Unmarshal([]byte(`{"instructions":42}`), &badInstr); err != nil {
		t.Fatal(err)
	}
	if _, eErr := badInstr.DecodeInstructions(); eErr == nil ||
		eErr.Class != errclass.ClassTranslation || eErr.Message != "instructions must be a string" {
		t.Errorf("bad instructions = %v", eErr)
	}

	var badIn ResponsesRequest
	if err := json.Unmarshal([]byte(`{"input":{"x":1}}`), &badIn); err != nil {
		t.Fatal(err)
	}
	if _, eErr := badIn.DecodeInputItems(); eErr == nil ||
		eErr.Class != errclass.ClassTranslation ||
		!strings.HasPrefix(eErr.Message, "input must be a string or an array of items: ") {
		t.Errorf("bad input = %v", eErr)
	}

	var arrayOut ResponsesRequest
	if err := json.Unmarshal([]byte(`{"input":[{"type":"function_call_output","call_id":"c","output":[{"type":"input_text","text":"a"},{"type":"input_text","text":"b"}]}]}`), &arrayOut); err != nil {
		t.Fatal(err)
	}
	items, eErr = arrayOut.DecodeInputItems()
	if eErr != nil || len(items) != 1 || items[0].Type != "function_call_output" {
		t.Fatalf("array output input = %+v, %v", items, eErr)
	}
	got, eErr := FunctionCallOutputText(items[0].Output)
	if eErr != nil || got != "ab" {
		t.Fatalf("array output text = %q, %v", got, eErr)
	}
}

func TestToolResultErrorPrefix(t *testing.T) {
	if ToolResultErrorPrefix != "[error] " {
		t.Fatalf("ToolResultErrorPrefix = %q, want %q", ToolResultErrorPrefix, "[error] ")
	}
	if got := ToolResultErrorPrefix + "boom"; got != "[error] boom" {
		t.Fatalf("prefix consumption = %q", got)
	}
}

// DecodeClaudeMessages owns Claude-request decoding for every adapter:
// envelope normalization (defaulted max tokens, flattened system,
// classified tool_choice) plus per-block records with pre-resolved image
// URLs; unknown types and roles are recorded verbatim for the target
// renderer to reject; all decode failures are ClassTranslation.
func TestDecodeClaudeMessages(t *testing.T) {
	rec, eErr := DecodeClaudeMessages([]byte(`{
		"max_tokens":256,"stream":true,"temperature":0.5,"top_p":0.9,
		"system":[{"type":"text","text":"s1"},{"type":"text","text":"s2"}],
		"stop_sequences":["END"],
		"tool_choice":{"type":"tool","name":"lookup"},
		"tools":[{"name":"f","input_schema":{"type":"object"}}],
		"thinking":{"type":"enabled","budget_tokens":1024},
		"messages":[
			{"role":"user","content":"hi"},
			{"role":"assistant","content":[
				{"type":"text","text":"hey"},
				{"type":"thinking","thinking":"hmm"},
				{"type":"redacted_thinking","data":"x"},
				{"type":"tool_use","id":"t1","name":"f","input":{"a":1}},
				{"type":"tool_use","id":"t2","name":"g"}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"t1","content":"ok","is_error":true},
				{"type":"image","source":{"type":"base64","data":"QUJD"}},
				{"type":"image","source":{"type":"url","url":"https://x/i.png"}}
			]},
			{"role":"robot","content":[{"type":"video"}]},
			{"role":"user","content":null}
		]
	}`))
	if eErr != nil {
		t.Fatalf("unexpected error: %v", eErr)
	}
	if rec.MaxTokens != 256 || !rec.Stream || rec.StopSequences == nil ||
		len(rec.StopSequences) != 1 || rec.StopSequences[0] != "END" {
		t.Fatalf("envelope = %+v", rec)
	}
	if rec.System != "s1\n\ns2" {
		t.Fatalf("system = %q", rec.System)
	}
	if rec.ToolChoiceKind != ToolChoiceNamed || rec.ToolChoiceName != "lookup" {
		t.Fatalf("tool_choice = %s/%s", rec.ToolChoiceKind, rec.ToolChoiceName)
	}
	if rec.Thinking == nil || rec.Thinking.BudgetTokens != 1024 ||
		rec.Temperature == nil || *rec.Temperature != 0.5 || rec.TopP == nil || *rec.TopP != 0.9 {
		t.Fatalf("optional envelope fields lost: %+v", rec)
	}
	if len(rec.Tools) != 1 || rec.Tools[0].Name != "f" {
		t.Fatalf("tools = %+v", rec.Tools)
	}
	if len(rec.Messages) != 5 {
		t.Fatalf("messages = %d", len(rec.Messages))
	}
	m0 := rec.Messages[0]
	if m0.Role != "user" || m0.Content != "hi" || m0.Blocks != nil {
		t.Fatalf("string-content record = %+v", m0)
	}
	a := rec.Messages[1]
	if a.Role != "assistant" || a.Content != "" || len(a.Blocks) != 5 {
		t.Fatalf("assistant record = %+v", a)
	}
	if a.Blocks[0].Kind != "text" || a.Blocks[0].Text != "hey" {
		t.Fatalf("text block = %+v", a.Blocks[0])
	}
	if a.Blocks[1].Kind != "thinking" || a.Blocks[2].Kind != "redacted_thinking" {
		t.Fatalf("thinking kinds = %+v %+v", a.Blocks[1], a.Blocks[2])
	}
	if a.Blocks[3].CallID != "t1" || a.Blocks[3].Name != "f" || string(a.Blocks[3].Input) != `{"a":1}` {
		t.Fatalf("tool_use block = %+v", a.Blocks[3])
	}
	if a.Blocks[4].CallID != "t2" || a.Blocks[4].Input != nil {
		t.Fatalf("absent-input tool_use = %+v", a.Blocks[4])
	}
	u := rec.Messages[2]
	if len(u.Blocks) != 3 {
		t.Fatalf("user blocks = %+v", u.Blocks)
	}
	tr := u.Blocks[0]
	if tr.Kind != "tool_result" || tr.CallID != "t1" || !tr.IsError || string(tr.Result) != `"ok"` {
		t.Fatalf("tool_result block = %+v", tr)
	}
	if img := u.Blocks[1]; img.Kind != "image" || img.URL != "data:application/octet-stream;base64,QUJD" {
		t.Fatalf("base64 image not resolved: %+v", img)
	}
	if img := u.Blocks[2]; img.URL != "https://x/i.png" {
		t.Fatalf("url image not resolved: %+v", img)
	}
	// Unknown role/type recorded verbatim — rejection names the TARGET
	// endpoint and belongs to the renderer.
	if r := rec.Messages[3]; r.Role != "robot" || len(r.Blocks) != 1 || r.Blocks[0].Kind != "video" {
		t.Fatalf("unknown role/type not recorded: %+v", r)
	}
	if null := rec.Messages[4]; null.Content != "" || null.Blocks != nil {
		t.Fatalf("null content must set neither field: %+v", null)
	}

	// Defaults: absent max_tokens → 4096, absent system/tool_choice quiet.
	rec, eErr = DecodeClaudeMessages([]byte(`{}`))
	if eErr != nil || rec.MaxTokens != 4096 || rec.System != "" ||
		rec.ToolChoiceKind != ToolChoiceAbsent || rec.Messages != nil {
		t.Fatalf("defaults = %+v, %v", rec, eErr)
	}

	for name, body := range map[string]string{
		"malformed json":         `{`,
		"malformed tool_choice":  `{"tool_choice":42}`,
		"unknown tool_choice":    `{"tool_choice":{"type":"blowup"}}`,
		"bad system":             `{"system":42}`,
		"system bad block":       `{"system":[{"type":"doc"}]}`,
		"content wrong shape":    `{"messages":[{"role":"user","content":42}]}`,
		"block not an object":    `{"messages":[{"role":"user","content":[42]}]}`,
		"block field wrong type": `{"messages":[{"role":"user","content":[{"type":"text","text":7}]}]}`,
		"image missing source":   `{"messages":[{"role":"user","content":[{"type":"image"}]}]}`,
		"url source missing url": `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url"}}]}]}`,
		"base64 source no data":  `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64"}}]}]}`,
		"unknown source type":    `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"s3"}}]}]}`,
	} {
		_, eErr := DecodeClaudeMessages([]byte(body))
		if eErr == nil || eErr.Class != errclass.ClassTranslation {
			t.Errorf("%s: want ClassTranslation, got %+v", name, eErr)
		}
	}
}
