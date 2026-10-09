package shared

import (
	"bytes"
	"encoding/json"
	"fmt"

	"ollama-cliproxyapi/internal/errclass"
)

func policyError(field, requirement string) *errclass.Error {
	return &errclass.Error{Class: errclass.ClassUnsupported, StatusCode: 400, Message: fmt.Sprintf("Responses option %s %s", field, requirement)}
}

// ValidateResponsesChatOptions validates only stateful or semantically
// unsupported options on the Responses -> Chat cross-format leg. Unlisted
// fields are intentionally ignored for forward compatibility.
func ValidateResponsesChatOptions(rawBody []byte) *errclass.Error {
	var fields map[string]json.RawMessage
	if json.Unmarshal(rawBody, &fields) != nil || fields == nil {
		return errclass.Translation("malformed Responses request JSON")
	}
	isNull := func(raw json.RawMessage) bool {
		return len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
	}
	boolFalse := func(name string) *errclass.Error {
		raw, ok := fields[name]
		if !ok || isNull(raw) {
			return nil
		}
		var v bool
		if json.Unmarshal(raw, &v) != nil {
			return policyError(name, "must be boolean or null")
		}
		if v {
			return policyError(name, "must be false")
		}
		return nil
	}
	for _, name := range []string{"store", "background"} {
		if e := boolFalse(name); e != nil {
			return e
		}
	}
	if raw, ok := fields["metadata"]; ok && !isNull(raw) {
		var v map[string]json.RawMessage
		if json.Unmarshal(raw, &v) != nil {
			return policyError("metadata", "must be an object or null")
		}
	}
	for _, name := range []string{"previous_response_id", "conversation"} {
		if raw, ok := fields[name]; ok && !isNull(raw) {
			var s string
			if json.Unmarshal(raw, &s) != nil {
				return policyError(name, "must be a string or null")
			}
			if s != "" {
				return policyError(name, "is stateful and unsupported")
			}
		}
	}
	if raw, ok := fields["include"]; ok && !isNull(raw) {
		var v []json.RawMessage
		if json.Unmarshal(raw, &v) != nil {
			return policyError("include", "must be an array or null")
		}
		if len(v) != 0 {
			return policyError("include", "must be empty")
		}
	}
	if raw, ok := fields["max_tool_calls"]; ok && !isNull(raw) {
		return policyError("max_tool_calls", "cannot be enforced by Chat Completions")
	}
	for name, allowed := range map[string]map[string]bool{
		"service_tier": {"auto": true, "default": true}, "truncation": {"disabled": true},
	} {
		if raw, ok := fields[name]; ok && !isNull(raw) {
			var s string
			if json.Unmarshal(raw, &s) != nil {
				return policyError(name, "must be a string or null")
			}
			if !allowed[s] {
				return policyError(name, "has an unsupported value")
			}
		}
	}
	for _, name := range []string{"prompt_cache_key", "prompt_cache_retention", "user", "safety_identifier"} {
		if raw, ok := fields[name]; ok && !isNull(raw) {
			var s string
			if json.Unmarshal(raw, &s) != nil {
				return policyError(name, "must be a string or null")
			}
		}
	}
	return nil
}

// ResponsesTextFormatToChat maps Responses text.format to Chat
// response_format while preserving schema bytes (including large integers).
func ResponsesTextFormatToChat(raw json.RawMessage) (json.RawMessage, *errclass.Error) {
	if !HasContent(raw) {
		return nil, nil
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil || obj == nil {
		return nil, errclass.Translation("Responses text format must be an object")
	}
	var typ string
	if json.Unmarshal(obj["type"], &typ) != nil {
		return nil, errclass.Translation("Responses text format type is required")
	}
	switch typ {
	case "text", "json_object":
		return append(json.RawMessage(nil), raw...), nil
	case "json_schema":
		var name string
		if json.Unmarshal(obj["name"], &name) != nil || name == "" || !HasContent(obj["schema"]) {
			return nil, errclass.Translation("Responses json_schema format requires name and schema")
		}
		inner := map[string]json.RawMessage{"name": obj["name"], "schema": obj["schema"]}
		for _, key := range []string{"description", "strict"} {
			if v, ok := obj[key]; ok {
				inner[key] = v
			}
		}
		wrapped, _ := json.Marshal(map[string]any{"type": "json_schema", "json_schema": inner})
		return wrapped, nil
	default:
		return nil, unsupportedResponsesTool("unsupported Responses text format type")
	}
}

// ChatResponseFormatToResponses is the inverse mapping.
func ChatResponseFormatToResponses(raw json.RawMessage) (json.RawMessage, *errclass.Error) {
	if !HasContent(raw) {
		return nil, nil
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil || obj == nil {
		return nil, errclass.Translation("Chat response_format must be an object")
	}
	var typ string
	if json.Unmarshal(obj["type"], &typ) != nil {
		return nil, errclass.Translation("Chat response_format type is required")
	}
	if typ == "text" || typ == "json_object" {
		return append(json.RawMessage(nil), raw...), nil
	}
	if typ != "json_schema" {
		return nil, unsupportedResponsesTool("unsupported Chat response_format type")
	}
	var inner map[string]json.RawMessage
	if json.Unmarshal(obj["json_schema"], &inner) != nil || inner == nil {
		return nil, errclass.Translation("Chat json_schema response_format requires json_schema")
	}
	var name string
	if json.Unmarshal(inner["name"], &name) != nil || name == "" || !HasContent(inner["schema"]) {
		return nil, errclass.Translation("Chat json_schema response_format requires name and schema")
	}
	out := map[string]json.RawMessage{"type": json.RawMessage(`"json_schema"`), "name": inner["name"], "schema": inner["schema"]}
	for _, key := range []string{"description", "strict"} {
		if v, ok := inner[key]; ok {
			out[key] = v
		}
	}
	b, _ := json.Marshal(out)
	return b, nil
}
