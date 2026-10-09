package plugin

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"
)

// reasoningChatFrames is a CommandCode-shaped chat-completions stream: the
// thinking text arrives under reasoning mirrored by reasoning_details[].text,
// then under a bare reasoning string, then the answer text. Nothing carries
// the standard reasoning_content.
var reasoningChatFrames = []string{
	`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}` + "\n\n",
	`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":"We","reasoning_details":[{"type":"reasoning.text","text":"We","format":"unknown","index":0}]}}]}` + "\n\n",
	`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":" need"}}]}` + "\n\n",
	`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"2"}}]}` + "\n\n",
	`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
	`data: [DONE]` + "\n\n",
}

const reasoningCompletion = `{"id":"c1","object":"chat.completion","model":"glm-5.2",` +
	`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"2","reasoning":"We need"}}],` +
	`"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`

// TestE2EReasoningToResponsesClient covers the migrated path end to end for a
// /v1/responses client (the interface the pi agent uses): the upstream chat
// reasoning becomes a reasoning item announced at output index 0 with its
// summary events, closes before the message item, and the terminal output
// keeps reasoning first.
func TestE2EReasoningToResponsesClient(t *testing.T) {
	m, f, st, _ := newIntegrationManager(t)
	st.setChatFrames(reasoningChatFrames)
	resp, err := m.HandleCall("executor.execute_stream",
		execStreamReqBodyForKey("ollama/glm-5.2", "openai-response",
			[]byte(`{"model":"ollama/glm-5.2","input":"hi","stream":true}`), "down-reason", "sk-test-1"))
	if err != nil {
		t.Fatalf("execute_stream: %v", err)
	}
	if env := decodeEnv(t, resp); !env.OK {
		t.Fatalf("envelope = %s", resp)
	}
	m.bridge.WaitForInFlight(5 * time.Second)

	joined := strings.Join(emittedEvents(t, f), "")
	for _, want := range []string{
		`"type":"reasoning"`,
		`"status":"in_progress"`,
		`response.reasoning_summary_part.added`,
		`response.reasoning_summary_text.delta`,
		`"delta":"We"`,
		`"delta":" need"`,
		`response.reasoning_summary_text.done`,
		`response.reasoning_summary_part.done`,
		`"summary":[{"type":"summary_text","text":"We need"}]`,
		`response.completed`,
		`"type":"message"`,
		`"text":"2"`,
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("downstream stream missing %q:\n%s", want, joined)
		}
	}
	// The mirrored reasoning field must not double the summary text.
	if strings.Contains(joined, `"text":"WeWe"`) {
		t.Fatalf("mirrored reasoning double-counted:\n%s", joined)
	}
	reasoningAt := strings.Index(joined, `response.output_item.added`)
	messageAt := strings.Index(joined, `"type":"message"`)
	if reasoningAt < 0 || messageAt < 0 || reasoningAt > messageAt {
		t.Fatalf("reasoning item must be announced before the message item:\n%s", joined)
	}
	assertCleanStreamClose(t, f)
}

// TestE2EReasoningToClaudeClient covers the claude client leg: the same
// upstream stream yields a leading thinking block (no signature — the vendor
// sends none) that closes before the text block opens.
func TestE2EReasoningToClaudeClient(t *testing.T) {
	m, f, st, _ := newIntegrationManager(t)
	st.setChatFrames(reasoningChatFrames)
	resp, err := m.HandleCall("executor.execute_stream",
		execStreamReqBodyForKey("ollama/glm-5.2", "claude",
			[]byte(`{"model":"ollama/glm-5.2","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"hi"}]}`), "down-think", "sk-test-1"))
	if err != nil {
		t.Fatalf("execute_stream: %v", err)
	}
	if env := decodeEnv(t, resp); !env.OK {
		t.Fatalf("envelope = %s", resp)
	}
	m.bridge.WaitForInFlight(5 * time.Second)

	joined := strings.Join(emittedEvents(t, f), "")
	for _, want := range []string{
		`"type":"thinking"`,
		`"thinking":"We","type":"thinking_delta"`,
		`"thinking":" need","type":"thinking_delta"`,
		`"text":"2","type":"text_delta"`,
		`message_stop`,
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("claude stream missing %q:\n%s", want, joined)
		}
	}
	if strings.Contains(joined, "signature") {
		t.Fatalf("thinking block must carry no signature:\n%s", joined)
	}
	if thinkAt, textAt := strings.Index(joined, `"type":"thinking_delta"`), strings.Index(joined, `"type":"text_delta"`); thinkAt > textAt {
		t.Fatalf("thinking must precede the text block:\n%s", joined)
	}
	assertCleanStreamClose(t, f)
}

// TestE2EReasoningToOpenAIClient covers the passthrough leg: the vendor
// spellings are mirrored onto reasoning_content while the vendor fields stay
// intact.
func TestE2EReasoningToOpenAIClient(t *testing.T) {
	m, f, st, _ := newIntegrationManager(t)
	st.setChatFrames(reasoningChatFrames)
	resp, err := m.HandleCall("executor.execute_stream",
		execStreamReqBodyForKey("ollama/glm-5.2", "openai",
			[]byte(`{"model":"ollama/glm-5.2","messages":[{"role":"user","content":"hi"}],"stream":true}`), "down-cc", "sk-test-1"))
	if err != nil {
		t.Fatalf("execute_stream: %v", err)
	}
	if env := decodeEnv(t, resp); !env.OK {
		t.Fatalf("envelope = %s", resp)
	}
	m.bridge.WaitForInFlight(5 * time.Second)

	joined := strings.Join(emittedEvents(t, f), "")
	if !strings.Contains(joined, `"reasoning_content":"We"`) || !strings.Contains(joined, `"reasoning_content":" need"`) {
		t.Fatalf("reasoning_content not backfilled:\n%s", joined)
	}
	if !strings.Contains(joined, `"reasoning_details":[{"type":"reasoning.text","text":"We","format":"unknown","index":0}]`) {
		t.Fatalf("vendor reasoning_details altered or dropped:\n%s", joined)
	}
	assertCleanStreamClose(t, f)
}

// TestE2EReasoningNonStream covers the non-stream legs for all three client
// formats off one upstream body.
func TestE2EReasoningNonStream(t *testing.T) {
	ccBody := []byte(`{"model":"ollama/glm-5.2","messages":[{"role":"user","content":"hi"}]}`)
	cases := []struct {
		name   string
		format string
		check  func(t *testing.T, result map[string]any)
	}{
		{"responses", "openai-response", func(t *testing.T, result map[string]any) {
			output := result["output"].([]any)
			if len(output) != 2 {
				t.Fatalf("output = %v, want reasoning + message", output)
			}
			rs := output[0].(map[string]any)
			if rs["type"] != "reasoning" || rs["id"] != "rs_c1_0" {
				t.Fatalf("reasoning item wrong: %v", rs)
			}
			if got := rs["summary"].([]any)[0].(map[string]any)["text"]; got != "We need" {
				t.Fatalf("summary text = %q", got)
			}
			if output[1].(map[string]any)["type"] != "message" {
				t.Fatalf("message item must follow: %v", output[1])
			}
		}},
		{"claude", "claude", func(t *testing.T, result map[string]any) {
			blocks := result["content"].([]any)
			if len(blocks) != 2 {
				t.Fatalf("blocks = %v, want thinking + text", blocks)
			}
			if think := blocks[0].(map[string]any); think["type"] != "thinking" || think["thinking"] != "We need" {
				t.Fatalf("thinking block wrong: %v", think)
			}
			if blocks[1].(map[string]any)["type"] != "text" {
				t.Fatalf("text block must follow: %v", blocks[1])
			}
		}},
		{"openai", "openai", func(t *testing.T, result map[string]any) {
			msg := result["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
			if msg["reasoning_content"] != "We need" || msg["reasoning"] != "We need" {
				t.Fatalf("message reasoning not backfilled: %v", msg)
			}
			if msg["content"] != "2" {
				t.Fatalf("content altered: %v", msg)
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, _, st, _ := newIntegrationManager(t)
			st.setChatBody(reasoningCompletion)
			var out pluginapi.ExecutorResponse
			decodeResult(t, mustHandle(t, m, "executor.execute", execReqBody("ollama/glm-5.2", tc.format, ccBody, false)), &out)
			var result map[string]any
			if err := json.Unmarshal(out.Payload, &result); err != nil {
				t.Fatalf("decode result: %v (%s)", err, out.Payload)
			}
			tc.check(t, result)
		})
	}
}
