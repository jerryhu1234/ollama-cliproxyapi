package chatcompletions

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

func TestResponsesCustomToolRequestAndMixedHistory(t *testing.T) {
	body := []byte(`{"model":"x","tools":[{"type":"custom","name":"exec","description":"run","format":{"type":"text"}},{"type":"function","name":"sum","parameters":{"type":"object"}}],"tool_choice":{"type":"custom","name":"exec"},"input":[{"type":"custom_tool_call","call_id":"c1","name":"exec","input":"a\n\"b\\c☃"},{"type":"function_call","call_id":"c2","name":"sum","arguments":"{\"x\":1}"},{"type":"custom_tool_call_output","call_id":"c1","output":[{"type":"input_text","text":"ok"}]},{"type":"function_call_output","call_id":"c2","output":"2"}]}`)
	out, e := BuildRequest("m", "openai-response", body, nil)
	if e != nil {
		t.Fatal(e)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	tools := got["tools"].([]any)
	params := tools[0].(map[string]any)["function"].(map[string]any)["parameters"].(map[string]any)
	if params["type"] != "object" || params["required"].([]any)[0] != "input" {
		t.Fatalf("custom schema = %#v", params)
	}
	msgs := got["messages"].([]any)
	calls := msgs[0].(map[string]any)["tool_calls"].([]any)
	if len(calls) != 2 {
		t.Fatalf("mixed calls = %#v", calls)
	}
	if args := calls[0].(map[string]any)["function"].(map[string]any)["arguments"].(string); args != `{"input":"a\n\"b\\c☃"}` {
		t.Fatalf("wrapped input = %q", args)
	}
	if len(msgs) != 3 || msgs[1].(map[string]any)["tool_call_id"] != "c1" || msgs[2].(map[string]any)["tool_call_id"] != "c2" {
		t.Fatalf("outputs = %#v", msgs)
	}
}

func TestResponsesCustomToolFormatAndInvalidNamespaceRejected(t *testing.T) {
	for _, body := range []string{
		`{"tools":[{"type":"custom","name":"x","format":{"type":"grammar","syntax":"lark"}}]}`,
		`{"tools":[{"type":"custom","name":"x","format":{}}]}`,
		`{"tools":[{"type":"namespace","name":"n","tools":[{"type":"namespace","name":"nested"}]}]}`,
		`{"tools":[{"type":"namespace","name":"a","tools":[{"type":"function","name":"x"}]},{"type":"namespace","name":"b","tools":[{"type":"function","name":"x"}]}],"input":[{"type":"function_call","name":"x","arguments":"{}"}]}`,
	} {
		if _, e := BuildRequest("m", "openai-response", []byte(body), nil, "strict"); e == nil || e.StatusCode != 400 {
			t.Fatalf("body %s error = %#v", body, e)
		}
	}
}

func TestResponsesNamespaceAdditionalToolsChoiceFormatStrictGolden(t *testing.T) {
	body := []byte(`{"model":"x","stream":true,"stream_options":{"include_usage":false},"text":{"format":{"type":"json_schema","name":"answer","strict":false,"schema":{"type":"object","properties":{"n":{"const":900719925474099312345}}}}},"tools":[{"type":"namespace","name":"fs","tools":[{"type":"custom","name":"patch","format":{"type":"text"}}]}],"tool_choice":{"type":"custom","namespace":"fs","name":"patch"},"input":[{"type":"additional_tools","tools":[{"type":"function","name":"sum","strict":true,"parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}]},{"type":"custom_tool_call","namespace":"fs","name":"patch","call_id":"c","input":"go"},{"type":"custom_tool_call_output","call_id":"c","output":"done"}]}`)
	out, e := BuildRequest("m", "openai-response", body, nil)
	if e != nil {
		t.Fatal(e)
	}
	s := string(out)
	for _, want := range []string{`"name":"fs__patch"`, `"name":"sum"`, `"strict":true`, `"name":"answer"`, `900719925474099312345`, `"include_usage":false`, `"tool_call_id":"c"`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
	if strings.Count(s, `"role":"assistant"`) != 1 {
		t.Fatalf("additional_tools became a message: %s", s)
	}
}

func TestResponsesOptionsPolicyOnlyCrossFormat(t *testing.T) {
	if _, e := BuildRequest("m", "openai-response", []byte(`{"store":true,"input":"x"}`), nil, "strict"); e == nil {
		t.Fatal("stateful Responses option accepted")
	}
	native := []byte(`{"model":"x","store":true,"messages":[]}`)
	out, e := BuildRequest("m", "openai", native, nil)
	if e != nil || !strings.Contains(string(out), `"store":true`) {
		t.Fatalf("native Chat changed: %s %v", out, e)
	}
}

func TestCPACompatibilityBuiltinGrammarAndStatefulDrop(t *testing.T) {
	cases := []struct {
		name, body     string
		wants, forbids []string
	}{
		{"builtin only", `{"input":"hi","tools":[{"type":"code_interpreter","container":{"type":"auto"}}],"tool_choice":"required"}`, []string{`"content":"hi"`}, []string{`"tools"`, `"tool_choice"`}},
		{"mixed", `{"input":"hi","tools":[{"type":"code_interpreter"},{"type":"function","name":"sum","parameters":{"type":"object"}},{"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark"}}],"tool_choice":"required"}`, []string{`"name":"sum"`, `"name":"apply_patch"`, `"tool_choice":"required"`}, []string{`code_interpreter`, `grammar`}},
		{"stateful ignored", `{"input":"hi","store":true,"background":true,"conversation":"c","previous_response_id":"p","service_tier":"priority","truncation":"auto","max_tool_calls":3,"include":["reasoning.encrypted_content"],"prompt_cache_key":"k"}`, []string{`"content":"hi"`}, []string{`store`, `background`, `conversation`, `previous_response_id`, `service_tier`, `truncation`, `max_tool_calls`, `include`, `prompt_cache_key`}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, e := BuildRequest("qwen3.8-max", "openai-response", []byte(tc.body), nil)
			if e != nil {
				t.Fatal(e)
			}
			s := string(out)
			for _, w := range tc.wants {
				if !strings.Contains(s, w) {
					t.Fatalf("missing %s: %s", w, s)
				}
			}
			for _, w := range tc.forbids {
				if strings.Contains(s, w) {
					t.Fatalf("unexpected %s: %s", w, s)
				}
			}
		})
	}
	out, e := BuildRequest("m", "openai-response", []byte(`{"input":"hi","tools":[{"type":"custom","name":"apply_patch","format":{"type":"grammar"}}]}`), nil, "strict")
	if e == nil || out != nil {
		t.Fatalf("strict accepted grammar: %s %v", out, e)
	}
}

func TestCPACompatibilityAdditionalToolsChoiceFiltering(t *testing.T) {
	cases := []struct {
		name, body            string
		wantTools, wantChoice bool
	}{
		{"additional builtin required", `{"input":[{"type":"additional_tools","tools":[{"type":"code_interpreter"}]},{"type":"message","role":"user","content":"hi"}],"tool_choice":"required"}`, false, false},
		{"additional builtin named", `{"input":[{"type":"additional_tools","tools":[{"type":"code_interpreter"}]},{"type":"message","role":"user","content":"hi"}],"tool_choice":{"type":"code_interpreter"}}`, false, false},
		{"additional mixed", `{"input":[{"type":"additional_tools","tools":[{"type":"code_interpreter"},{"type":"function","name":"sum","parameters":{"type":"object"}}]},{"type":"message","role":"user","content":"hi"}],"tool_choice":"required"}`, true, true},
		{"top additional mixed", `{"tools":[{"type":"code_interpreter"}],"input":[{"type":"additional_tools","tools":[{"type":"function","name":"sum","parameters":{"type":"object"}}]},{"type":"message","role":"user","content":"hi"}],"tool_choice":"required"}`, true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, e := BuildRequest("m", "openai-response", []byte(tc.body), nil)
			if e != nil {
				t.Fatal(e)
			}
			var got map[string]json.RawMessage
			if json.Unmarshal(out, &got) != nil {
				t.Fatal(string(out))
			}
			_, hasTools := got["tools"]
			_, hasChoice := got["tool_choice"]
			if hasTools != tc.wantTools || hasChoice != tc.wantChoice {
				t.Fatalf("tools=%t choice=%t output=%s", hasTools, hasChoice, out)
			}
			if bytes.Contains(out, []byte("code_interpreter")) {
				t.Fatalf("builtin leaked: %s", out)
			}
		})
	}
	secret := `secret-additional-token`
	_, e := BuildRequest("m", "openai-response", []byte(`{"input":[{"type":"additional_tools","tools":[{"type":"code_interpreter","container":{"secret":"`+secret+`"}}]}]}`), nil, "strict")
	if e == nil || strings.Contains(e.Message, secret) {
		t.Fatalf("strict error leaked or missing: %#v", e)
	}
}

func TestNamespaceResponseRecoveryNonStreamAndStream(t *testing.T) {
	req := []byte(`{"tools":[{"type":"namespace","name":"fs","tools":[{"type":"custom","name":"patch"},{"type":"function","name":"stat"}]}]}`)
	upstream := []byte(`{"id":"r","model":"m","choices":[{"message":{"tool_calls":[{"id":"c","function":{"name":"fs__patch","arguments":"{\"input\":\"go\"}"}},{"id":"f","function":{"name":"fs__stat","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
	out, e := ConvertNonStreamResponse("openai-response", 200, upstream, req)
	if e != nil {
		t.Fatal(e)
	}
	for _, want := range []string{`"type":"custom_tool_call"`, `"name":"patch"`, `"namespace":"fs"`, `"type":"function_call"`, `"name":"stat"`} {
		if !strings.Contains(string(out), want) {
			t.Fatalf("missing %s in %s", want, out)
		}
	}

	sc := NewStreamConverter("openai-response", req)
	frames := []string{
		`{"id":"r","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"fs__pa","arguments":"{\"in"}},{"index":1,"id":"f","function":{"name":"fs__stat","arguments":"{}"}}]},"finish_reason":""}]}`,
		`{"id":"r","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"tch","arguments":"put\":\"go\"}"}}]},"finish_reason":"tool_calls"}]}`,
	}
	var joined strings.Builder
	for _, frame := range frames {
		events, _, err := sc.Feed([]byte("data: " + frame + "\n\n"))
		if err != nil {
			t.Fatal(err)
		}
		for _, event := range events {
			joined.Write(event)
		}
	}
	events, _, err := sc.Feed([]byte("data: [DONE]\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range events {
		joined.Write(event)
	}
	s := joined.String()
	for _, want := range []string{`"name":"patch"`, `"namespace":"fs"`, `"input":"go"`, `"name":"stat"`, `"arguments":"{}"`, `response.completed`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
	if strings.Contains(s, `"name":"fs__patch"`) {
		t.Fatalf("flattened name leaked downstream: %s", s)
	}
}

func TestNamespaceRecoveryDoesNotJoinByEmptyOrDuplicateCallID(t *testing.T) {
	req := []byte(`{"tools":[{"type":"namespace","name":"left","tools":[{"type":"function","name":"same"}]},{"type":"namespace","name":"right","tools":[{"type":"function","name":"same"}]}]}`)
	for _, calls := range []string{
		`[{"id":"","function":{"name":"left__same","arguments":"{\"side\":\"l\"}"}},{"id":"","function":{"name":"right__same","arguments":"{\"side\":\"r\"}"}}]`,
		`[{"id":"dup","function":{"name":"left__same","arguments":"{\"side\":\"l\"}"}},{"id":"dup","function":{"name":"right__same","arguments":"{\"side\":\"r\"}"}}]`,
	} {
		upstream := []byte(`{"id":"r","model":"m","choices":[{"message":{"tool_calls":` + calls + `},"finish_reason":"tool_calls"}]}`)
		out, e := ConvertNonStreamResponse("openai-response", 200, upstream, req)
		if e != nil {
			t.Fatal(e)
		}
		var response struct {
			Output []struct{ Name, Namespace, Arguments string } `json:"output"`
		}
		if json.Unmarshal(out, &response) != nil || len(response.Output) != 2 {
			t.Fatalf("output=%s", out)
		}
		if response.Output[0].Name != "same" || response.Output[0].Namespace != "left" || !strings.Contains(response.Output[0].Arguments, `"l"`) {
			t.Fatalf("left identity crossed: %s", out)
		}
		if response.Output[1].Name != "same" || response.Output[1].Namespace != "right" || !strings.Contains(response.Output[1].Arguments, `"r"`) {
			t.Fatalf("right identity crossed: %s", out)
		}
	}

	// Streaming terminal output must bind namespace to each tool index, not the
	// repeated call_id. Added and completed items retain the same identities.
	sc := NewStreamConverter("openai-response", req)
	frame := `{"id":"r","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"dup","function":{"name":"left__same","arguments":"{\"side\":\"l\"}"}},{"index":1,"id":"dup","function":{"name":"right__same","arguments":"{\"side\":\"r\"}"}}]},"finish_reason":"tool_calls"}]}`
	events, _, e := sc.Feed([]byte("data: " + frame + "\n\ndata: [DONE]\n\n"))
	if e != nil {
		t.Fatal(e)
	}
	var joined strings.Builder
	for _, event := range events {
		joined.Write(event)
	}
	s := joined.String()
	if strings.Count(s, `"namespace":"left"`) < 2 || strings.Count(s, `"namespace":"right"`) < 2 {
		t.Fatalf("stream namespace crossed or missing: %s", s)
	}
}

func TestFunctionStreamNameAndPendingArgumentFragments(t *testing.T) {
	request := []byte(`{"tools":[{"type":"function","name":"sum"},{"type":"function","name":"summary"},{"type":"custom","name":"exec"}]}`)
	tests := []struct {
		name   string
		frames []string
	}{
		{
			name: "fragmented name with early args",
			frames: []string{
				`{"id":"r","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"s","arguments":"{\"x\":"}}]},"finish_reason":""}]}`,
				`{"id":"r","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"um","arguments":"1}"}}]},"finish_reason":"tool_calls"}]}`,
			},
		},
		{
			name: "arguments before name",
			frames: []string{
				`{"id":"r","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"arguments":"{\"x\":"}}]},"finish_reason":""}]}`,
				`{"id":"r","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"sum","arguments":"1}"}}]},"finish_reason":"tool_calls"}]}`,
			},
		},
		{
			name: "repeated complete name",
			frames: []string{
				`{"id":"r","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"sum","arguments":"{\"x\":"}}]},"finish_reason":""}]}`,
				`{"id":"r","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"sum","arguments":"1}"}}]},"finish_reason":"tool_calls"}]}`,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sc := NewStreamConverter("openai-response", request)
			var joined strings.Builder
			for _, frame := range tc.frames {
				events, _, e := sc.Feed([]byte("data: " + frame + "\n\n"))
				if e != nil {
					t.Fatal(e)
				}
				for _, event := range events {
					joined.Write(event)
				}
			}
			events, _, e := sc.Feed([]byte("data: [DONE]\n\n"))
			if e != nil {
				t.Fatal(e)
			}
			for _, event := range events {
				joined.Write(event)
			}
			s := joined.String()
			if strings.Count(s, `"arguments":"","call_id":"c","name":"sum","type":"function_call"`) != 1 {
				t.Fatalf("opening item name changed or duplicated: %s", s)
			}
			if strings.Contains(s, `"name":"s"`) || strings.Contains(s, `"name":"sumsum"`) {
				t.Fatalf("partial/repeated name announced: %s", s)
			}
			var deltas strings.Builder
			for _, line := range strings.Split(s, "\n") {
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				var raw map[string]any
				if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &raw) != nil {
					continue
				}
				if raw["type"] == "response.function_call_arguments.delta" && raw["item_id"] == "c" {
					deltas.WriteString(raw["delta"].(string))
				}
			}
			if deltas.String() != `{"x":1}` {
				t.Fatalf("pending/current argument bytes = %q, want complete ordered arguments; stream=%s", deltas.String(), s)
			}
			if !strings.Contains(s, `"type":"function_call","call_id":"c","name":"sum","arguments":"{\"x\":1}"`) {
				t.Fatalf("completed output mismatch: %s", s)
			}
		})
	}
}

func TestMixedCustomFunctionStreamInterleavingKeepsIdentity(t *testing.T) {
	req := []byte(`{"tools":[{"type":"function","name":"sum"},{"type":"custom","name":"exec"}]}`)
	sc := NewStreamConverter("openai-response", req)
	frames := []string{
		`{"id":"r","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"f","function":{"arguments":"{\"x\":"}},{"index":1,"id":"c","function":{"name":"ex","arguments":"{\"in"}}]},"finish_reason":""}]}`,
		`{"id":"r","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"name":"ec","arguments":"put\":\"go\"}"}},{"index":0,"function":{"name":"s","arguments":"1"}}]},"finish_reason":""}]}`,
		`{"id":"r","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"um","arguments":"}"}}]},"finish_reason":"tool_calls"}]}`,
	}
	var joined strings.Builder
	for _, frame := range frames {
		events, _, e := sc.Feed([]byte("data: " + frame + "\n\n"))
		if e != nil {
			t.Fatal(e)
		}
		for _, event := range events {
			joined.Write(event)
		}
	}
	events, _, e := sc.Feed([]byte("data: [DONE]\n\n"))
	if e != nil {
		t.Fatal(e)
	}
	for _, event := range events {
		joined.Write(event)
	}
	s := joined.String()
	for _, want := range []string{
		`"output_index":0`, `"call_id":"f"`, `"name":"sum"`, `"arguments":"{\"x\":1}"`,
		`"output_index":1`, `"call_id":"c"`, `"name":"exec"`, `"input":"go"`,
		`response.custom_tool_call_input.done`, `response.output_item.done`, `response.completed`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
	if strings.Contains(s, `"item_id":"c","output_index":1,"type":"response.function_call_arguments.delta"`) {
		t.Fatalf("custom args emitted as function delta: %s", s)
	}
}

func TestCustomResponseUsesOnlyRequestProvenance(t *testing.T) {
	upstream := []byte(`{"id":"r","model":"m","choices":[{"message":{"tool_calls":[{"id":"c","type":"function","function":{"name":"same","arguments":"{\"input\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}`)
	customReq := []byte(`{"tools":[{"type":"custom","name":"same"}]}`)
	functionReq := []byte(`{"tools":[{"type":"function","name":"same","parameters":{"type":"object"}}]}`)
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			out, e := ConvertNonStreamResponse("openai-response", 200, upstream, customReq)
			if e != nil || !strings.Contains(string(out), `"type":"custom_tool_call"`) || !strings.Contains(string(out), `"input":"pwd"`) {
				t.Errorf("custom: %s %#v", out, e)
			}
		}()
		go func() {
			defer wg.Done()
			out, e := ConvertNonStreamResponse("openai-response", 200, upstream, functionReq)
			if e != nil || !strings.Contains(string(out), `"type":"function_call"`) || !strings.Contains(string(out), `"arguments":"{\"input\":\"pwd\"}"`) {
				t.Errorf("function: %s %#v", out, e)
			}
		}()
	}
	wg.Wait()
	out, e := ConvertNonStreamResponse("openai-response", 200, upstream)
	if e != nil || !strings.Contains(string(out), `"type":"function_call"`) {
		t.Fatalf("no original request must be conservative: %s %#v", out, e)
	}
}

func TestCustomResponseMalformedWrapperIsSafe(t *testing.T) {
	req := []byte(`{"tools":[{"type":"custom","name":"exec"}]}`)
	for _, args := range []string{`not-json-secret`, `{"input":7}`, `{}`} {
		body, _ := json.Marshal(map[string]any{
			"id": "r",
			"choices": []any{map[string]any{
				"message": map[string]any{"tool_calls": []any{map[string]any{
					"id": "c", "function": map[string]any{"name": "exec", "arguments": args},
				}}},
			}},
		})
		_, e := ConvertNonStreamResponse("openai-response", 200, body, req)
		if e == nil || strings.Contains(e.Message, "secret") || strings.Contains(e.Message, args) {
			t.Fatalf("unsafe error for %q: %#v", args, e)
		}
	}
}

func TestCustomStreamFragmentedAndParallel(t *testing.T) {
	req := []byte(`{"tools":[{"type":"custom","name":"exec"},{"type":"function","name":"sum"}]}`)
	sc := NewStreamConverter("openai-response", req)
	frames := []string{
		`data: {"id":"r","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"ex","arguments":"{\"in"}},{"index":1,"id":"c1","function":{"name":"sum","arguments":"{\"x\":"}}]},"finish_reason":""}]}` + "\n\n",
		`data: {"id":"r","model":"m","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"1}"}},{"index":0,"function":{"name":"ec","arguments":"put\":\"a\\n\\\"b\\\\c☃\"}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n",
		`data: [DONE]` + "\n\n",
	}
	var all strings.Builder
	for _, frame := range frames {
		for _, piece := range []string{frame[:len(frame)/3], frame[len(frame)/3 : 2*len(frame)/3], frame[2*len(frame)/3:]} {
			events, _, e := sc.Feed([]byte(piece))
			if e != nil {
				t.Fatal(e)
			}
			for _, event := range events {
				all.Write(event)
			}
		}
	}
	s := all.String()
	for _, want := range []string{`response.custom_tool_call_input.done`, `"type":"custom_tool_call"`, `"input":"a\n\"b\\c☃"`, `"type":"function_call"`, `"arguments":"{\"x\":1}"`, `response.completed`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
	if strings.Contains(s, `"item_id":"c0","output_index":0,"type":"response.function_call_arguments.delta"`) {
		t.Fatalf("custom fragments leaked as function deltas: %s", s)
	}
}
