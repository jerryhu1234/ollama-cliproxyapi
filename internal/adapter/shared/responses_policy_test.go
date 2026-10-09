package shared

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

func TestBuildResponsesToolContextNamespaceAdditionalStrict(t *testing.T) {
	strict := true
	req := &ResponsesRequest{
		Tools: []RespTool{{Type: "namespace", Name: "fs", Tools: []RespTool{{Type: "custom", Name: "patch", Format: json.RawMessage(`{"type":"text"}`)}}}},
		Input: json.RawMessage(`[{"type":"additional_tools","tools":[{"type":"function","name":"sum","strict":true,"parameters":{"type":"object","properties":{"n":{"const":900719925474099312345}}}}]}]`),
	}
	ctx, tools, e := BuildResponsesToolContext(req, "chat")
	if e != nil || len(tools) != 2 {
		t.Fatalf("tools=%#v err=%v", tools, e)
	}
	if id, e := ctx.ResolveCall("fs", "patch"); e != nil || id.ChatName != "fs__patch" || id.Kind != "custom" {
		t.Fatalf("namespace identity=%#v err=%v", id, e)
	}
	if id, ok := ctx.ResolveChatName("sum"); !ok || id.Kind != "function" {
		t.Fatalf("reverse=%#v %v", id, ok)
	}
	if tools[1].Function.Strict == nil || *tools[1].Function.Strict != strict || !bytes.Contains(tools[1].Function.Parameters, []byte("900719925474099312345")) {
		t.Fatalf("strict/schema lost: %#v", tools[1])
	}
}

func TestBuildResponsesToolContextRejectsAmbiguityCollisionAndUnsupported(t *testing.T) {
	for name, req := range map[string]*ResponsesRequest{
		"collision": {Tools: []RespTool{{Type: "function", Name: "a__b"}, {Type: "namespace", Name: "a", Tools: []RespTool{{Type: "function", Name: "b"}}}}},
		"nested":    {Tools: []RespTool{{Type: "namespace", Name: "a", Tools: []RespTool{{Type: "namespace", Name: "b"}}}}},
		"builtin":   {Tools: []RespTool{{Type: "web_search", Name: "web"}}},
		"long":      {Tools: []RespTool{{Type: "namespace", Name: strings.Repeat("n", 63), Tools: []RespTool{{Type: "function", Name: "xx"}}}}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, e := BuildResponsesToolContext(req, "chat"); e == nil {
				t.Fatal("expected rejection")
			}
		})
	}
	req := &ResponsesRequest{Tools: []RespTool{{Type: "namespace", Name: "a", Tools: []RespTool{{Type: "function", Name: "same"}}}, {Type: "namespace", Name: "b", Tools: []RespTool{{Type: "custom", Name: "same"}}}}}
	ctx, _, e := BuildResponsesToolContext(req, "chat")
	if e != nil {
		t.Fatal(e)
	}
	if _, e = ctx.ResolveCall("", "same"); e == nil {
		t.Fatal("ambiguous bare local accepted")
	}
}

func TestBuildResponsesToolContextRejectsDuplicateIdentityWithDifferentKind(t *testing.T) {
	function := RespTool{Type: "function", Name: "same", Description: "d", Strict: boolPtr(true), Parameters: json.RawMessage(`{"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}`)}
	custom := RespTool{Type: "custom", Name: "same", Description: "d", Strict: boolPtr(true)}
	for _, tools := range [][]RespTool{{function, custom}, {custom, function}} {
		req := &ResponsesRequest{Tools: []RespTool{{Type: "namespace", Name: "n", Tools: tools}}}
		if _, _, e := BuildResponsesToolContext(req, "chat"); e == nil || e.StatusCode != 400 {
			t.Fatalf("different kinds merged for order %#v: %v", tools, e)
		}
	}
}

func boolPtr(v bool) *bool { return &v }

func TestResponsesFormatRoundTripPreservesRawSchema(t *testing.T) {
	raw := json.RawMessage(`{"type":"json_schema","name":"n","description":"d","strict":false,"schema":{"type":"object","properties":{"n":{"const":900719925474099312345}}}}`)
	chat, e := ResponsesTextFormatToChat(raw)
	if e != nil {
		t.Fatal(e)
	}
	back, e := ChatResponseFormatToResponses(chat)
	if e != nil {
		t.Fatal(e)
	}
	if !bytes.Contains(chat, []byte("900719925474099312345")) || !bytes.Contains(back, []byte("900719925474099312345")) || !bytes.Contains(back, []byte(`"strict":false`)) {
		t.Fatalf("roundtrip lost fields: %s %s", chat, back)
	}
	for _, bad := range []json.RawMessage{json.RawMessage(`{"type":"unknown"}`), json.RawMessage(`{"type":"json_schema","schema":{}}`)} {
		if _, e := ResponsesTextFormatToChat(bad); e == nil {
			t.Fatalf("accepted %s", bad)
		}
	}
}

func TestValidateResponsesChatOptionsMatrixAndRedaction(t *testing.T) {
	allowed := `{"metadata":{},"store":false,"background":null,"service_tier":"default","truncation":"disabled","prompt_cache_key":"k","prompt_cache_retention":"24h","user":"u","safety_identifier":"s","include":[],"unknown_future":{"x":1}}`
	if e := ValidateResponsesChatOptions([]byte(allowed)); e != nil {
		t.Fatal(e)
	}
	for _, bad := range []string{
		`{"store":true}`, `{"background":true}`, `{"previous_response_id":"secret-id"}`, `{"conversation":"secret-conv"}`, `{"include":["secret"]}`, `{"max_tool_calls":2}`, `{"truncation":"auto"}`, `{"service_tier":"priority"}`, `{"metadata":"bad"}`,
	} {
		e := ValidateResponsesChatOptions([]byte(bad))
		if e == nil {
			t.Fatalf("accepted %s", bad)
		}
		if strings.Contains(e.Message, "secret") {
			t.Fatalf("leaked value: %s", e.Message)
		}
	}
}

func TestResponsesToolContextRequestLocalConcurrency(t *testing.T) {
	custom := &ResponsesRequest{Tools: []RespTool{{Type: "custom", Name: "same"}}}
	function := &ResponsesRequest{Tools: []RespTool{{Type: "function", Name: "same"}}}
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			c, _, e := BuildResponsesToolContext(custom, "chat")
			if e != nil || !c.IsCustom("same") {
				t.Errorf("custom context %v %v", c, e)
			}
		}()
		go func() {
			defer wg.Done()
			c, _, e := BuildResponsesToolContext(function, "chat")
			if e != nil || c.IsCustom("same") {
				t.Errorf("function context %v %v", c, e)
			}
		}()
	}
	wg.Wait()
}
