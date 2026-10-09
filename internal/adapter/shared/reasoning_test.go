package shared

import (
	"encoding/json"
	"testing"
)

func TestReasoningTextPriority(t *testing.T) {
	cases := []struct {
		name   string
		fields ReasoningFields
		want   string
		ok     bool
	}{
		{
			// Vendors mirror the token in both spellings; details-only must
			// win so the text is not duplicated.
			name:   "details outrank the plain string",
			fields: ReasoningFields{Reasoning: "We need", ReasoningDetails: []ReasoningDetail{{Text: "We need"}}},
			want:   "We need", ok: true,
		},
		{
			name:   "details join in arrival order",
			fields: ReasoningFields{ReasoningDetails: []ReasoningDetail{{Text: "a"}, {Text: "b"}}},
			want:   "ab", ok: true,
		},
		{
			name:   "blank details fall through to reasoning",
			fields: ReasoningFields{Reasoning: "plain", ReasoningDetails: []ReasoningDetail{{Text: "  "}}},
			want:   "plain", ok: true,
		},
		{
			name:   "standard reasoning_content is the last resort",
			fields: ReasoningFields{ReasoningContent: "std"},
			want:   "std", ok: true,
		},
		{
			name:   "no thinking at all",
			fields: ReasoningFields{},
			want:   "", ok: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tc.fields.ReasoningText()
			if got != tc.want || ok != tc.ok {
				t.Fatalf("ReasoningText() = (%q, %v), want (%q, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

func TestBackfillReasoningContentDelta(t *testing.T) {
	t.Run("reasoning string is mirrored", func(t *testing.T) {
		body := []byte(`{"id":"c1","created":1789639872,"system_fingerprint":"fp_1","choices":[{"index":0,"delta":{"reasoning":"We"},"finish_reason":null}]}`)
		out := BackfillReasoningContent(body, "delta")
		if out == nil {
			t.Fatal("expected a rewritten body")
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatalf("output not JSON: %v", err)
		}
		delta := m["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
		if delta["reasoning_content"] != "We" || delta["reasoning"] != "We" {
			t.Fatalf("backfill wrong: %v", delta)
		}
		// Untouched members keep their original wire representation.
		if string(out) == string(body) || !containsAll(string(out),
			`"created":1789639872`, `"system_fingerprint":"fp_1"`, `"reasoning":"We"`) {
			t.Fatalf("original members not preserved verbatim: %s", out)
		}
	})

	t.Run("details only", func(t *testing.T) {
		body := []byte(`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"We"},{"text":" need"}]}}]}`)
		out := BackfillReasoningContent(body, "delta")
		if out == nil || !containsAll(string(out), `"reasoning_content":"We need"`) {
			t.Fatalf("details text not backfilled: %s", out)
		}
	})

	t.Run("existing reasoning_content is left alone", func(t *testing.T) {
		body := []byte(`{"choices":[{"delta":{"reasoning":"vendor","reasoning_content":"standard"}}]}`)
		if out := BackfillReasoningContent(body, "delta"); out != nil {
			t.Fatalf("standard member already present, want unchanged: %s", out)
		}
	})

	t.Run("no reasoning member", func(t *testing.T) {
		body := []byte(`{"choices":[{"delta":{"content":"2"}}]}`)
		if out := BackfillReasoningContent(body, "delta"); out != nil {
			t.Fatalf("plain content chunk must stay byte-identical, got %s", out)
		}
	})

	t.Run("invalid json stays untouched", func(t *testing.T) {
		if out := BackfillReasoningContent([]byte(`{"choices":[{"delta":{"reasoning":`), "delta"); out != nil {
			t.Fatalf("malformed body must not be rewritten: %s", out)
		}
	})

	t.Run("message field for non-stream bodies", func(t *testing.T) {
		body := []byte(`{"choices":[{"message":{"role":"assistant","content":"2","reasoning":"We need 2"}}]}`)
		out := BackfillReasoningContent(body, "message")
		if out == nil || !containsAll(string(out), `"reasoning_content":"We need 2"`) {
			t.Fatalf("message reasoning not backfilled: %s", out)
		}
		if got := BackfillReasoningContent(body, "delta"); got != nil {
			t.Fatalf("delta field absent, want unchanged: %s", got)
		}
	})

	t.Run("every choice is filled", func(t *testing.T) {
		body := []byte(`{"choices":[{"delta":{"reasoning":"a"}},{"delta":{"reasoning_content":"x"}},{"delta":{"reasoning":"c"}}]}`)
		out := BackfillReasoningContent(body, "delta")
		if out == nil || !containsAll(string(out), `"reasoning_content":"a"`, `"reasoning_content":"c"`) {
			t.Fatalf("choices not all filled: %s", out)
		}
	})
}

func TestReasoningItemShapes(t *testing.T) {
	item := NewRespReasoningItem(ReasoningItemID("r1", 0), "thought")
	raw, err := json.Marshal(item)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"id":"rs_r1_0","type":"reasoning","encrypted_content":"","summary":[{"type":"summary_text","text":"thought"}]}`
	if string(raw) != want {
		t.Fatalf("reasoning item = %s, want %s", raw, want)
	}

	added, err := json.Marshal(ReasoningItemInProgress("rs_r1_0"))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(added, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["id"] != "rs_r1_0" || got["type"] != "reasoning" || got["status"] != "in_progress" {
		t.Fatalf("in-progress item = %s", added)
	}
	if summary, ok := got["summary"].([]any); !ok || len(summary) != 0 {
		t.Fatalf("in-progress summary must be an empty array: %s", added)
	}

	block, err := json.Marshal(ClaudeThinkingBlock("thought"))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(block) != `{"thinking":"thought","type":"thinking"}` {
		t.Fatalf("thinking block = %s", block)
	}
}

func TestOutputAssemblerReasoningLeads(t *testing.T) {
	oa := NewOutputAssembler("r1")
	oa.AppendReasoning(NewRespReasoningItem(ReasoningItemID("r1", 0), "thought"))
	oa.ReserveTextSlot()
	oa.AppendFunctionCall("c1", "f", `{"k":1}`)
	oa.AddText("answer")

	out := oa.Render()
	if len(out) != 3 {
		t.Fatalf("want reasoning + message + function_call, got %v", out)
	}
	if item, ok := out[0].(RespReasoningItem); !ok || item.Type != "reasoning" || item.Summary[0].Text != "thought" {
		t.Fatalf("reasoning item must lead the output: %v", out[0])
	}
	msg := out[1].(RespItem)
	if msg.Type != "message" || string(msg.Content) != `[{"type":"output_text","text":"answer"}]` {
		t.Fatalf("message item wrong: %v", msg)
	}
	if fc := out[2].(RespItem); fc.Type != "function_call" || fc.CallID != "c1" {
		t.Fatalf("function_call item wrong: %v", fc)
	}
}

func containsAll(haystack string, needles ...string) bool {
	for _, n := range needles {
		if !contains(haystack, n) {
			return false
		}
	}
	return true
}

func contains(haystack, needle string) bool {
	return len(needle) == 0 || len(haystack) >= len(needle) && indexOf(haystack, needle) >= 0
}

func indexOf(haystack, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}
