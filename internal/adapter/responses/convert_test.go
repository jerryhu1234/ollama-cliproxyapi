package responses

import (
	"encoding/json"
	"strings"
	"testing"

	"ollama-cliproxyapi/internal/errclass"
)

// ---- ConvertNonStreamResponse (FR-006, AC §D) --------------------------

const respTextOnly = `{"id":"resp_t","object":"response","status":"completed","model":"gpt-5.6-luna",` +
	`"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi there"}]}],` +
	`"usage":{"input_tokens":5,"output_tokens":6,"total_tokens":11}}`

const respFunctionCall = `{"id":"resp_f","object":"response","status":"completed","model":"gpt-5.6-luna",` +
	`"output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"sf\"}"}],` +
	`"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}`

const respIncomplete = `{"id":"resp_i","object":"response","status":"incomplete","model":"gpt-5.6-luna",` +
	`"output":[],"usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1}}`

func TestConvertResponsesStatusErrors(t *testing.T) {
	_, eErr := ConvertNonStreamResponse("openai", 500, []byte("upstream exploded"))
	if eErr == nil || eErr.Class != errclass.ClassUpstream || !eErr.Retryable {
		t.Fatalf("err = %+v", eErr)
	}

	// Long bodies are bounded to a redacted snippet, never echoed whole.
	_, eErr = ConvertNonStreamResponse("openai", 500, []byte(strings.Repeat("z", 200)))
	if eErr == nil || !strings.HasSuffix(eErr.Message, "...") || len(eErr.Message) > 83 {
		t.Fatalf("upstream body not bounded: %q", eErr.Message)
	}
}

func TestConvertResponsesPassthrough(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai-response", 200, []byte(respTextOnly))
	if eErr != nil || string(out) != respTextOnly {
		t.Fatalf("passthrough = %s %v", out, eErr)
	}
	if _, eErr := ConvertNonStreamResponse("openai-response", 200, []byte("{bad")); eErr == nil ||
		eErr.Class != errclass.ClassTranslation {
		t.Fatalf("malformed passthrough = %v", eErr)
	}
}

func TestConvertResponsesToChatTextOnly(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(respTextOnly))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var cc struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Model   string `json:"model"`
		Created int64  `json:"created"`
		Choices []struct {
			Message struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []any  `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int64 `json:"prompt_tokens"`
			CompletionTokens int64 `json:"completion_tokens"`
			TotalTokens      int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(out, &cc); err != nil {
		t.Fatalf("not chat.completion: %v (%s)", err, out)
	}
	if cc.ID != "resp_t" || cc.Object != "chat.completion" || cc.Model != "gpt-5.6-luna" || cc.Created == 0 {
		t.Fatalf("identity fields wrong: %+v", cc)
	}
	c0 := cc.Choices[0]
	if c0.Message.Role != "assistant" || c0.Message.Content != "hi there" ||
		c0.Message.ToolCalls != nil || c0.FinishReason != "stop" {
		t.Fatalf("choice wrong: %+v", c0)
	}
	if cc.Usage.PromptTokens != 5 || cc.Usage.CompletionTokens != 6 || cc.Usage.TotalTokens != 11 {
		t.Fatalf("usage mapping wrong: %+v", cc.Usage)
	}
}

func TestConvertResponsesToChatFunctionCall(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(respFunctionCall))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var cc struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(out, &cc); err != nil {
		t.Fatalf("decode: %v", err)
	}
	tc := cc.Choices[0].Message.ToolCalls
	if cc.Choices[0].FinishReason != "tool_calls" || len(tc) != 1 ||
		tc[0].ID != "call_1" || tc[0].Type != "function" ||
		tc[0].Function.Name != "get_weather" ||
		!strings.Contains(tc[0].Function.Arguments, `"sf"`) {
		t.Fatalf("function_call not mapped: %s", out)
	}

	// Absent arguments decode to an empty call.
	absent := strings.Replace(respFunctionCall, `"arguments":"{\"city\":\"sf\"}"`, `"arguments":""`, 1)
	out, eErr = ConvertNonStreamResponse("openai", 200, []byte(absent))
	if eErr != nil {
		t.Fatalf("absent args convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"arguments":"{}"`) {
		t.Fatalf("absent arguments not defaulted: %s", out)
	}
}

func TestConvertResponsesToChatIncompleteLength(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(respIncomplete))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"finish_reason":"length"`) {
		t.Fatalf("incomplete not mapped to length: %s", out)
	}
}

func TestConvertResponsesToChatIncompleteWithToolsKeepsToolCalls(t *testing.T) {
	// Shared precedence: incomplete cannot downgrade tool_calls (matches
	// the stream path).
	body := strings.Replace(respFunctionCall, `"status":"completed"`, `"status":"incomplete"`, 1)
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"finish_reason":"tool_calls"`) {
		t.Fatalf("incomplete downgraded tool_calls: %s", out)
	}
}

func TestConvertResponsesToChatTotalTokensComputed(t *testing.T) {
	// Upstream omits total_tokens; the mapper must not emit 0 beside
	// nonzero counts (matches the sibling mappers' computed sum).
	body := strings.Replace(respTextOnly, `,"total_tokens":11`, "", 1)
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var cc struct {
		Usage struct {
			TotalTokens int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(out, &cc); err != nil {
		t.Fatalf("decode: %v (%s)", err, out)
	}
	if cc.Usage.TotalTokens != 11 {
		t.Fatalf("total_tokens not computed: %s", out)
	}
}

func TestConvertResponsesToClaudeTextOnly(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("claude", 200, []byte(respTextOnly))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var msg struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Model   string `json:"model"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(out, &msg); err != nil {
		t.Fatalf("not Messages response: %v (%s)", err, out)
	}
	if msg.ID != "resp_t" || msg.Type != "message" || msg.Role != "assistant" ||
		msg.Model != "gpt-5.6-luna" || msg.StopReason != "end_turn" {
		t.Fatalf("identity wrong: %+v", msg)
	}
	if len(msg.Content) != 1 || msg.Content[0].Type != "text" || msg.Content[0].Text != "hi there" {
		t.Fatalf("text block wrong: %s", out)
	}
	if msg.Usage.InputTokens != 5 || msg.Usage.OutputTokens != 6 {
		t.Fatalf("usage wrong: %+v", msg.Usage)
	}
}

// Canonical Claude-result kernel pins shared with the Chat Completions
// route: stop_sequence is omitted entirely (never null) and empty
// output_text parts ship no dead text block.
func TestConvertResponsesToClaudeCanonicalShape(t *testing.T) {
	body := strings.Replace(respTextOnly,
		`[{"type":"output_text","text":"hi there"}]`,
		`[{"type":"output_text","text":""},{"type":"output_text","text":"hi there"}]`, 1)
	out, eErr := ConvertNonStreamResponse("claude", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	s := string(out)
	if strings.Contains(s, `"stop_sequence"`) || strings.Contains(s, `"text":""`) {
		t.Fatalf("non-canonical claude shape: %s", out)
	}
}

func TestConvertResponsesToClaudeFunctionCall(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("claude", 200, []byte(respFunctionCall))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var msg struct {
		Content []struct {
			Type  string         `json:"type"`
			ID    string         `json:"id"`
			Name  string         `json:"name"`
			Input map[string]any `json:"input"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
	}
	if err := json.Unmarshal(out, &msg); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if msg.StopReason != "tool_use" || len(msg.Content) != 1 ||
		msg.Content[0].Type != "tool_use" || msg.Content[0].ID != "call_1" ||
		msg.Content[0].Name != "get_weather" || msg.Content[0].Input["city"] != "sf" {
		t.Fatalf("tool_use not mapped: %s", out)
	}

	// Malformed arguments are a translation failure, never dropped.
	bad := strings.Replace(respFunctionCall, `"arguments":"{\"city\":\"sf\"}"`, `"arguments":"nope"`, 1)
	_, eErr = ConvertNonStreamResponse("claude", 200, []byte(bad))
	if eErr == nil || !strings.Contains(eErr.Message, "malformed tool call arguments JSON") {
		t.Fatalf("malformed args err = %v", eErr)
	}
}

func TestConvertResponsesToClaudeIncompleteMaxTokens(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("claude", 200, []byte(respIncomplete))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"stop_reason":"max_tokens"`) {
		t.Fatalf("incomplete not mapped to max_tokens: %s", out)
	}
}

func TestConvertResponsesToClaudeIncompleteWithToolsKeepsToolUse(t *testing.T) {
	// Shared precedence: incomplete cannot downgrade tool_use (matches the
	// stream path and the Chat Completions mapper).
	body := strings.Replace(respFunctionCall, `"status":"completed"`, `"status":"incomplete"`, 1)
	out, eErr := ConvertNonStreamResponse("claude", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"stop_reason":"tool_use"`) {
		t.Fatalf("incomplete downgraded tool_use: %s", out)
	}
}

func TestConvertResponsesUnknownFormat(t *testing.T) {
	_, eErr := ConvertNonStreamResponse("grpc", 200, []byte("{}"))
	if eErr == nil || eErr.Class != errclass.ClassUnsupported {
		t.Fatalf("err = %v", eErr)
	}
}

func TestConvertResponsesMalformedUpstream(t *testing.T) {
	for _, format := range []string{"openai", "claude"} {
		_, eErr := ConvertNonStreamResponse(format, 200, []byte("{not json"))
		if eErr == nil || eErr.Class != errclass.ClassTranslation ||
			!strings.Contains(eErr.Message, "malformed Responses response JSON") {
			t.Fatalf("%s: err = %v", format, eErr)
		}
	}
}

func TestConvertResponsesNoOutputItems(t *testing.T) {
	_, eErr := ConvertNonStreamResponse("openai", 200, []byte(`{"id":"x"}`))
	if eErr == nil || !strings.Contains(eErr.Message, "carries no output items") {
		t.Fatalf("err = %v", eErr)
	}
}

func TestConvertResponsesUnknownOutputItem(t *testing.T) {
	// An unrecognized item carrying real content (message parts) must
	// still fail descriptively rather than be dropped.
	body := strings.Replace(respTextOnly,
		`"output":[{"type":"message"`,
		`"output":[{"type":"audio"`, 1)
	for _, format := range []string{"openai", "claude"} {
		_, eErr := ConvertNonStreamResponse(format, 200, []byte(body))
		if eErr == nil || eErr.Class != errclass.ClassTranslation ||
			!strings.Contains(eErr.Message, `"audio"`) {
			t.Fatalf("%s: err = %v", format, eErr)
		}
	}
}

// Reasoning output items (emitted whenever Luna reasons) are informational
// and are omitted per the FR-005/FR-006 compatibility policy — matching
// the stream side — instead of failing the whole conversion.
func TestConvertResponsesReasoningItemOmitted(t *testing.T) {
	body := strings.Replace(respTextOnly,
		`"output":[{"type":"message"`,
		`"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"pondering"}]},{"type":"message"`, 1)
	for _, format := range []string{"openai", "claude"} {
		out, eErr := ConvertNonStreamResponse(format, 200, []byte(body))
		if eErr != nil {
			t.Fatalf("%s: reasoning item errored: %v", format, eErr)
		}
		if !strings.Contains(string(out), `"hi there"`) ||
			strings.Contains(string(out), "pondering") {
			t.Fatalf("%s: reasoning item not omitted cleanly: %s", format, out)
		}
	}
}

func TestConvertResponsesUsageDetailsAndAbsentUsage(t *testing.T) {
	body := strings.Replace(respTextOnly, `"usage":{"input_tokens":5,"output_tokens":6,"total_tokens":11}`, `"usage":{"input_tokens":10,"output_tokens":4,"input_tokens_details":{"cached_tokens":7,"cache_write_tokens":2},"output_tokens_details":{"reasoning_tokens":3}}`, 1)
	for _, format := range []string{"openai", "claude"} {
		out, eErr := ConvertNonStreamResponse(format, 200, []byte(body))
		if eErr != nil {
			t.Fatalf("%s: %v", format, eErr)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		u := m["usage"].(map[string]any)
		if format == "openai" {
			if u["prompt_tokens"] != float64(10) || u["prompt_tokens_details"].(map[string]any)["cached_tokens"] != float64(7) || u["completion_tokens_details"].(map[string]any)["reasoning_tokens"] != float64(3) {
				t.Fatalf("chat usage = %v", u)
			}
		} else if u["input_tokens"] != float64(1) || u["cache_read_input_tokens"] != float64(7) || u["cache_creation_input_tokens"] != float64(2) {
			t.Fatalf("claude usage = %v", u)
		}
	}
	if _, eErr := ConvertNonStreamResponse("openai", 200, []byte(strings.Replace(respTextOnly, `,"usage":{"input_tokens":5,"output_tokens":6,"total_tokens":11}`, "", 1))); eErr != nil {
		t.Fatalf("absent usage must remain valid: %v", eErr)
	}
}

// A bare unknown item with no decodable payload is informational too.
func TestConvertResponsesPayloadlessUnknownItemOmitted(t *testing.T) {
	body := strings.Replace(respTextOnly,
		`"output":[{"type":"message"`,
		`"output":[{"type":"web_search_call"},{"type":"message"`, 1)
	for _, format := range []string{"openai", "claude"} {
		out, eErr := ConvertNonStreamResponse(format, 200, []byte(body))
		if eErr != nil {
			t.Fatalf("%s: informational item errored: %v", format, eErr)
		}
		if !strings.Contains(string(out), `"hi there"`) {
			t.Fatalf("%s: message lost: %s", format, out)
		}
	}
}
