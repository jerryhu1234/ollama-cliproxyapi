package responses

import (
	"encoding/json"
	"strings"
	"testing"

	"ollama-cliproxyapi/internal/errclass"
)

func TestChatResponseFormatGolden(t *testing.T) {
	const large = `900719925474099312345678901234567890`
	tests := []struct {
		name   string
		format string
		want   string
	}{
		{"text", `{"type":"text"}`, `{"model":"upstream","text":{"format":{"type":"text"}}}`},
		{"json object", `{"type":"json_object"}`, `{"model":"upstream","text":{"format":{"type":"json_object"}}}`},
		{"json schema strict false", `{"type":"json_schema","json_schema":{"name":"answer","description":"d","strict":false,"schema":{"type":"object","properties":{"id":{"const":` + large + `}}}}}`, `{"model":"upstream","text":{"format":{"description":"d","name":"answer","schema":{"type":"object","properties":{"id":{"const":` + large + `}}},"strict":false,"type":"json_schema"}}}`},
		{"json schema strict true", `{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object"}}}`, `{"model":"upstream","text":{"format":{"name":"answer","schema":{"type":"object"},"strict":true,"type":"json_schema"}}}`},
		{"json schema strict absent", `{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"}}}`, `{"model":"upstream","text":{"format":{"name":"answer","schema":{"type":"object"},"type":"json_schema"}}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"messages":[],"response_format":` + tt.format + `}`)
			got := string(mustBuild(t, "upstream", "openai", body, nil))
			if got != tt.want {
				t.Fatalf("output mismatch\n got: %s\nwant: %s", got, tt.want)
			}
		})
	}
}

func TestChatResponseFormatErrorsAreSafe(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"unknown", `{"messages":[],"response_format":{"type":"private_kind","secret_schema_key":"do-not-reflect"}}`},
		{"missing name", `{"messages":[],"response_format":{"type":"json_schema","json_schema":{"schema":{"secret_schema_key":true}}}}`},
		{"missing schema", `{"messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"do-not-reflect"}}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, eErr := BuildRequest("m", "openai", []byte(tt.body), nil)
			if eErr == nil {
				t.Fatal("expected error")
			}
			if strings.Contains(eErr.Message, "private_kind") || strings.Contains(eErr.Message, "secret_schema_key") || strings.Contains(eErr.Message, "do-not-reflect") {
				t.Fatalf("error reflected request details: %q", eErr.Message)
			}
		})
	}
}

func TestChatFunctionStrictPassesThrough(t *testing.T) {
	for _, strict := range []string{"false", "true"} {
		body := []byte(`{"messages":[],"tools":[{"type":"function","function":{"name":"f","strict":` + strict + `}}]}`)
		m := decodeReq(t, mustBuild(t, "m", "openai", body, nil))
		tool := m["tools"].([]any)[0].(map[string]any)
		if got := tool["strict"]; got != (strict == "true") {
			t.Fatalf("strict %s became %v", strict, got)
		}
	}
}

func TestChatNPolicy(t *testing.T) {
	for _, body := range []string{`{"messages":[],"n":2}`, `{"messages":[],"n":1.5}`, `{"messages":[],"n":"2"}`, `{"messages":[],"n":0}`} {
		_, eErr := BuildRequest("m", "openai", []byte(body), nil)
		if eErr == nil {
			t.Fatalf("expected rejection for %s", body)
		}
		if body == `{"messages":[],"n":2}` && eErr.Class != errclass.ClassUnsupported {
			t.Fatalf("n=2 class = %s", eErr.Class)
		}
	}
	for _, body := range []string{`{"messages":[],"n":1}`, `{"messages":[]}`} {
		mustBuild(t, "m", "openai", []byte(body), nil)
	}
}

func TestNativeResponsesHighRiskFieldsPassThrough(t *testing.T) {
	body := []byte(`{"model":"client","input":"hi","previous_response_id":"resp_1","background":true,"tools":[{"type":"web_search_preview"}],"include":["web_search_call.action.sources"],"text":{"format":{"type":"json_schema","name":"x","schema":{"const":900719925474099312345678901234567890}}}}`)
	out := mustBuild(t, "upstream", "openai-response", body, nil)
	var got map[string]json.RawMessage
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	var original map[string]json.RawMessage
	_ = json.Unmarshal(body, &original)
	for _, key := range []string{"input", "previous_response_id", "background", "tools", "include", "text"} {
		if string(got[key]) != string(original[key]) {
			t.Errorf("%s changed: got %s want %s", key, got[key], original[key])
		}
	}
	if string(got["model"]) != `"upstream"` {
		t.Errorf("model = %s", got["model"])
	}
}
