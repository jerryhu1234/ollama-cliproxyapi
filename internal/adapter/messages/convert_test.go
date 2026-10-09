package messages

import (
	"encoding/json"
	"strings"
	"testing"

	"ollama-cliproxyapi/internal/errclass"
)

// ---- ConvertNonStreamResponse (FR-006, AC §C) --------------------------

const claudeTextOnly = `{"id":"m_1","type":"message","role":"assistant","model":"qwen3.7-max",` +
	`"content":[{"type":"text","text":"hello world"}],"stop_reason":"end_turn",` +
	`"usage":{"input_tokens":5,"output_tokens":7}}`

const claudeToolUse = `{"id":"m_2","type":"message","role":"assistant","model":"qwen3.7-max",` +
	`"content":[{"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"sf"}}],` +
	`"stop_reason":"tool_use","usage":{"input_tokens":9,"output_tokens":4}}`

const claudeThinking = `{"id":"m_3","type":"message","role":"assistant","model":"qwen3.7-max",` +
	`"content":[{"type":"thinking","thinking":"hmm","signature":"s1"},{"type":"text","text":"answer"}],` +
	`"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}`

func TestConvertMessagesStatusErrors(t *testing.T) {
	_, eErr := ConvertNonStreamResponse("openai", 503, []byte("upstream down"))
	if eErr == nil || eErr.Class != errclass.ClassUpstream || !eErr.Retryable {
		t.Fatalf("err = %+v", eErr)
	}
	if strings.Contains(eErr.Message, "sk-") {
		t.Fatalf("status error leaks body: %q", eErr.Message)
	}

	// Long bodies are bounded to a redacted snippet, never echoed whole.
	_, eErr = ConvertNonStreamResponse("openai", 500, []byte(strings.Repeat("x", 200)))
	if eErr == nil || !strings.HasSuffix(eErr.Message, "...") || len(eErr.Message) > 83 {
		t.Fatalf("upstream body not bounded: %q", eErr.Message)
	}

	// Redaction work is bounded: bodies beyond 4096 bytes are truncated
	// before snippet extraction, so the message stays snippet-sized.
	_, eErr = ConvertNonStreamResponse("openai", 500, []byte(strings.Repeat("x", 1<<20)))
	if eErr == nil || len(eErr.Message) > 83 {
		t.Fatalf("oversized upstream body not bounded: len = %d", len(eErr.Message))
	}
}

func TestConvertMessagesClaudePassthrough(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("claude", 200, []byte(claudeTextOnly))
	if eErr != nil || string(out) != claudeTextOnly {
		t.Fatalf("passthrough = %s %v", out, eErr)
	}
	if _, eErr := ConvertNonStreamResponse("claude", 200, []byte("{bad")); eErr == nil ||
		eErr.Class != errclass.ClassTranslation {
		t.Fatalf("malformed passthrough = %v", eErr)
	}
}

func TestConvertMessagesToOpenAITextOnly(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(claudeTextOnly))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var cc struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Model   string `json:"model"`
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
		Created int64 `json:"created"`
	}
	if err := json.Unmarshal(out, &cc); err != nil {
		t.Fatalf("not chat.completion: %v (%s)", err, out)
	}
	if cc.ID != "m_1" || cc.Object != "chat.completion" || cc.Model != "qwen3.7-max" || cc.Created == 0 {
		t.Fatalf("identity fields wrong: %+v", cc)
	}
	c0 := cc.Choices[0]
	if c0.Message.Role != "assistant" || c0.Message.Content != "hello world" ||
		c0.Message.ToolCalls != nil || c0.FinishReason != "stop" {
		t.Fatalf("choice wrong: %+v", c0)
	}
	if cc.Usage.PromptTokens != 5 || cc.Usage.CompletionTokens != 7 || cc.Usage.TotalTokens != 12 {
		t.Fatalf("usage mapping wrong: %+v", cc.Usage)
	}
}

func TestConvertMessagesToOpenAIToolUse(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(claudeToolUse))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var cc struct {
		Choices []struct {
			Message struct {
				Content   any `json:"content"`
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
		tc[0].ID != "tu_1" || tc[0].Type != "function" ||
		tc[0].Function.Name != "get_weather" ||
		!strings.Contains(tc[0].Function.Arguments, `"sf"`) {
		t.Fatalf("tool_use not mapped: %s", out)
	}
}

func TestConvertMessagesToOpenAIMaxTokensLength(t *testing.T) {
	body := strings.Replace(claudeTextOnly, `"stop_reason":"end_turn"`, `"stop_reason":"max_tokens"`, 1)
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"finish_reason":"length"`) {
		t.Fatalf("max_tokens not mapped: %s", out)
	}
}

func TestConvertMessagesMaxTokensWithToolsKeepsToolCalls(t *testing.T) {
	// Observed tool calls outrank the status-derived reason: a terminal
	// max_tokens must not downgrade finish_reason to length (FR-006).
	body := strings.Replace(claudeToolUse, `"stop_reason":"tool_use"`, `"stop_reason":"max_tokens"`, 1)
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"finish_reason":"tool_calls"`) {
		t.Fatalf("tool calls downgraded by max_tokens: %s", out)
	}
}

func TestConvertMessagesToResponsesMaxTokensIncomplete(t *testing.T) {
	body := strings.Replace(claudeTextOnly, `"stop_reason":"end_turn"`, `"stop_reason":"max_tokens"`, 1)
	out, eErr := ConvertNonStreamResponse("openai-response", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"status":"incomplete"`) {
		t.Fatalf("max_tokens not mapped to incomplete: %s", out)
	}
}

func TestConvertMessagesToResponsesMaxTokensWithToolsIncomplete(t *testing.T) {
	// Status derives from stop reason only: max_tokens marks even a
	// tool-call turn incomplete; function_call items carry the tool
	// signal (FR-006).
	body := strings.Replace(claudeToolUse, `"stop_reason":"tool_use"`, `"stop_reason":"max_tokens"`, 1)
	out, eErr := ConvertNonStreamResponse("openai-response", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"status":"incomplete"`) ||
		!strings.Contains(string(out), `"type":"function_call"`) {
		t.Fatalf("max_tokens with tools not incomplete+function_call: %s", out)
	}
}

func TestConvertMessagesToOpenAIRefusalContentFilter(t *testing.T) {
	// Shared stop table: refusal maps to content_filter on the
	// non-stream path too (previously diverged to stop).
	body := strings.Replace(claudeTextOnly, `"stop_reason":"end_turn"`, `"stop_reason":"refusal"`, 1)
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"finish_reason":"content_filter"`) {
		t.Fatalf("refusal not mapped: %s", out)
	}
}

func TestConvertMessagesToOpenAIThinkingOmitted(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(claudeThinking))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var cc struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []any  `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(out, &cc); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if cc.Choices[0].Message.Content != "answer" || cc.Choices[0].Message.ToolCalls != nil {
		t.Fatalf("thinking block leaked or text lost: %s", out)
	}
	if strings.Contains(string(out), "thinking") {
		t.Fatalf("reasoning material in output: %s", out)
	}
}

func TestConvertMessagesToResponsesTextOnly(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai-response", 200, []byte(claudeTextOnly))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var resp struct {
		ID     string `json:"id"`
		Object string `json:"object"`
		Model  string `json:"model"`
		Status string `json:"status"`
		Output []struct {
			Type    string `json:"type"`
			ID      string `json:"id"`
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
			TotalTokens  int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatalf("not Responses result: %v (%s)", err, out)
	}
	if resp.ID != "m_1" || resp.Object != "response" || resp.Model != "qwen3.7-max" || resp.Status != "completed" {
		t.Fatalf("identity wrong: %+v", resp)
	}
	if len(resp.Output) != 1 || resp.Output[0].Type != "message" ||
		resp.Output[0].ID != "m_1" ||
		resp.Output[0].Content[0].Type != "output_text" ||
		resp.Output[0].Content[0].Text != "hello world" {
		t.Fatalf("output_text aggregation wrong: %s", out)
	}
	if resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 7 || resp.Usage.TotalTokens != 12 {
		t.Fatalf("usage wrong: %+v", resp.Usage)
	}
}

func TestConvertMessagesToResponsesToolUse(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai-response", 200, []byte(claudeToolUse))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var resp struct {
		Output []struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"output"`
	}
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Output) != 1 || resp.Output[0].Type != "function_call" ||
		resp.Output[0].CallID != "tu_1" || resp.Output[0].Name != "get_weather" ||
		!strings.Contains(resp.Output[0].Arguments, `"sf"`) {
		t.Fatalf("function_call not mapped: %s", out)
	}
}

func TestConvertMessagesToResponsesBlockOrderMatchesStream(t *testing.T) {
	// Non-stream synthesis must emit items in upstream block order like
	// the streaming synthesizer (outputItems): [text, tool_use] yields
	// [message, function_call], not the post-loop append order
	// (FR-006 ordering parity across modes).
	body := `{"id":"m_6","type":"message","role":"assistant","model":"qwen3.7-max",` +
		`"content":[{"type":"text","text":"doing it"},{"type":"tool_use","id":"tu_2","name":"run","input":{"x":1}}],` +
		`"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
	out, eErr := ConvertNonStreamResponse("openai-response", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var resp struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			CallID string `json:"call_id"`
		} `json:"output"`
	}
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatalf("decode: %v (%s)", err, out)
	}
	if len(resp.Output) != 2 ||
		resp.Output[0].Type != "message" || resp.Output[0].Content[0].Text != "doing it" ||
		resp.Output[1].Type != "function_call" || resp.Output[1].CallID != "tu_2" {
		t.Fatalf("block order diverges from stream path: %s", out)
	}
}

func TestConvertMessagesUnknownFormat(t *testing.T) {
	_, eErr := ConvertNonStreamResponse("grpc", 200, []byte("{}"))
	if eErr == nil || eErr.Class != errclass.ClassUnsupported {
		t.Fatalf("err = %v", eErr)
	}
}

func TestConvertMessagesMalformedUpstream(t *testing.T) {
	for _, format := range []string{"openai", "openai-response"} {
		_, eErr := ConvertNonStreamResponse(format, 200, []byte("{not json"))
		if eErr == nil || eErr.Class != errclass.ClassTranslation ||
			!strings.Contains(eErr.Message, "malformed Messages response JSON") {
			t.Fatalf("%s: err = %v", format, eErr)
		}
	}
}

func TestConvertMessagesNoContentBlocks(t *testing.T) {
	_, eErr := ConvertNonStreamResponse("openai", 200, []byte(`{"id":"m_x","type":"message"}`))
	if eErr == nil || !strings.Contains(eErr.Message, "carries no content blocks") {
		t.Fatalf("err = %v", eErr)
	}
}

func TestConvertMessagesUnknownBlockType(t *testing.T) {
	body := strings.Replace(claudeTextOnly,
		`"content":[{"type":"text","text":"hello world"}]`,
		`"content":[{"type":"audio"}]`, 1)
	for _, format := range []string{"openai", "openai-response"} {
		_, eErr := ConvertNonStreamResponse(format, 200, []byte(body))
		if eErr == nil || eErr.Class != errclass.ClassTranslation ||
			!strings.Contains(eErr.Message, `"audio"`) {
			t.Fatalf("%s: err = %v", format, eErr)
		}
	}
}

func TestConvertMessagesToolUseAbsentInput(t *testing.T) {
	body := `{"id":"m_4","model":"qwen3.7-max","stop_reason":"tool_use",` +
		`"content":[{"type":"tool_use","id":"tu_9","name":"ping"}],` +
		`"usage":{"input_tokens":1,"output_tokens":1}}`
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"arguments":"{}"`) {
		t.Fatalf("absent input not defaulted: %s", out)
	}
}

func TestConvertMessagesToResponsesThinkingOmitted(t *testing.T) {
	out, eErr := ConvertNonStreamResponse("openai-response", 200, []byte(claudeThinking))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"text":"answer"`) || strings.Contains(string(out), "thinking") {
		t.Fatalf("thinking leaked or text lost: %s", out)
	}
}

// redacted_thinking is pure encrypted reasoning metadata with no
// representable payload; like thinking it is omitted (FR-005) instead of
// failing conversion, matching the stream path. Content-bearing unknown
// block types still error (see TestConvertMessagesUnknownBlockType).
func TestConvertMessagesRedactedThinkingOmitted(t *testing.T) {
	body := `{"id":"m_5","type":"message","role":"assistant","model":"qwen3.7-max",` +
		`"content":[{"type":"redacted_thinking","data":"encrypted-blob"},{"type":"text","text":"answer"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}`
	for _, format := range []string{"openai", "openai-response"} {
		out, eErr := ConvertNonStreamResponse(format, 200, []byte(body))
		if eErr != nil {
			t.Fatalf("%s: convert: %v", format, eErr)
		}
		if !strings.Contains(string(out), `"answer"`) ||
			strings.Contains(string(out), "redacted") || strings.Contains(string(out), "encrypted") {
			t.Fatalf("%s: redacted_thinking leaked or text lost: %s", format, out)
		}
	}
}

// Aggregated text ships verbatim (repo-wide verbatim policy): interior
// spacing across blocks survives untouched and whitespace-only aggregated
// content is preserved rather than dropped.
func TestConvertMessagesAggregatedTextVerbatim(t *testing.T) {
	body := `{"id":"m_7","type":"message","role":"assistant","model":"qwen3.7-max",` +
		`"content":[{"type":"text","text":"hello "},{"type":"text","text":" world"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	out, eErr := ConvertNonStreamResponse("openai", 200, []byte(body))
	if eErr != nil {
		t.Fatalf("convert: %v", eErr)
	}
	var cc struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(out, &cc); err != nil {
		t.Fatalf("decode: %v (%s)", err, out)
	}
	if got := cc.Choices[0].Message.Content; got != "hello  world" {
		t.Fatalf("content = %q, want verbatim %q", got, "hello  world")
	}

	ws := strings.Replace(claudeTextOnly, `"text":"hello world"`, `"text":"  \n\t "`, 1)
	out, eErr = ConvertNonStreamResponse("openai", 200, []byte(ws))
	if eErr != nil {
		t.Fatalf("whitespace convert: %v", eErr)
	}
	if !strings.Contains(string(out), `"content":"  \n\t "`) {
		t.Fatalf("whitespace-only aggregated text not shipped verbatim: %s", out)
	}
}

func TestConvertMessagesCacheUsageDetails(t *testing.T) {
	body := `{"id":"m","model":"m","content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":2,"cache_read_input_tokens":3,"cache_creation_input_tokens":1}}`
	for _, format := range []string{"openai", "openai-response"} {
		out, eErr := ConvertNonStreamResponse(format, 200, []byte(body))
		if eErr != nil {
			t.Fatalf("%s: %v", format, eErr)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		u := m["usage"].(map[string]any)
		if u["prompt_tokens"] != float64(8) && u["input_tokens"] != float64(8) {
			t.Fatalf("%s input = %v", format, u)
		}
		if format == "openai-response" && (u["input_tokens_details"].(map[string]any)["cached_tokens"] != float64(3) || u["input_tokens_details"].(map[string]any)["cache_write_tokens"] != float64(1)) {
			t.Fatalf("responses details = %v", u)
		}
	}
}
