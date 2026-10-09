// Package chatcompletions implements the Ollama Chat Completions
// protocol adapter (arch §8) for routes served via /v1/chat/completions:
// request conversion (FR-005), non-stream response conversion (FR-006),
// and SSE stream conversion (AC §B).
package chatcompletions

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"

	"ollama-cliproxyapi/internal/adapter/shared"
	"ollama-cliproxyapi/internal/catalog"
	"ollama-cliproxyapi/internal/errclass"
	"ollama-cliproxyapi/internal/thinking"
)

// EndpointPath is the upstream Ollama Chat Completions endpoint.
var EndpointPath = catalog.RouteChatCompletions.EndpointPath()

// AuthHeaders returns the FR-007 authorization headers for a Ollama key.
func AuthHeaders(key string) http.Header {
	h := http.Header{}
	h.Set("Authorization", "Bearer "+key)
	return h
}

// BuildRequest translates an inbound request body from sourceFormat
// ("openai" Chat Completions, "openai-response" Responses, "claude"
// Messages) into a Chat Completions body for upstreamModel (FR-005).
// ts carries the target model's thinking capability so reasoning
// budgets map to a supported effort level instead of being silently
// degraded (FR-005). Unknown formats are ClassUnsupported; malformed
// input is ClassTranslation. Errors are descriptive and redacted — no
// silent loss of tools or reasoning controls.
func BuildRequest(upstreamModel, sourceFormat string, sourceBody []byte, ts *pluginapi.ThinkingSupport, responsesCompatibility ...string) ([]byte, *errclass.Error) {
	switch sourceFormat {
	case "openai":
		return buildOpenAIRequest(upstreamModel, sourceBody)
	case "claude":
		return claudeToChat(upstreamModel, sourceBody, ts)
	case "openai-response":
		mode := "cpa"
		if len(responsesCompatibility) > 0 && responsesCompatibility[0] != "" {
			mode = responsesCompatibility[0]
		}
		return responsesToChat(upstreamModel, sourceBody, ts, mode)
	default:
		return nil, shared.UnsupportedFormat(sourceFormat, EndpointPath)
	}
}

// buildOpenAIRequest rewrites the top-level model field of an OpenAI request
// body to upstreamModel and strips any malformed top-level thinking object.
// DeepSeek models fail if thinking lacks a valid string type field.
func buildOpenAIRequest(upstreamModel string, body []byte) ([]byte, *errclass.Error) {
	var req map[string]json.RawMessage
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, errclass.Translation("malformed openai request JSON: " + err.Error())
	}
	if req == nil {
		return nil, errclass.Translation("malformed request body: JSON null is not a valid request")
	}
	rawThinking, hasThinking := req["thinking"]
	validThinking := hasThinking && isValidThinking(rawThinking)
	if hasThinking && !validThinking {
		delete(req, "thinking")
	}
	// Usage capture parity with responsesToChat/claudeToChat: when a
	// client already carries stream_options.include_usage the body is
	// returned unchanged (fast path below); when it streams without
	// stream_options, inject the usage trailer so upstream returns usage
	// (CPA reads manager usage/TPS from it) and skip the fast path since
	// the body changed.
	usageInjected := false
	if string(req["stream"]) == "true" {
		if raw, okOptions := req["stream_options"]; !okOptions || len(raw) == 0 || string(raw) == "null" {
			req["stream_options"] = json.RawMessage(`{"include_usage":true}`)
			usageInjected = true
		}
	}
	if !usageInjected {
		if raw, ok := req["model"]; ok && string(raw) == `"`+upstreamModel+`"` && (validThinking || !hasThinking) {
			return body, nil
		}
	}
	req["model"] = json.RawMessage(`"` + upstreamModel + `"`)
	b, err := json.Marshal(req)
	if err != nil {
		return nil, errclass.Translation("model id cannot be represented as JSON")
	}
	return b, nil
}

// isValidThinking reports whether raw is a JSON object with a non-empty string "type" field.
func isValidThinking(raw json.RawMessage) bool {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil || obj == nil {
		return false
	}
	rawType, ok := obj["type"]
	if !ok {
		return false
	}
	var typeStr string
	if err := json.Unmarshal(rawType, &typeStr); err != nil {
		return false
	}
	return strings.TrimSpace(typeStr) != ""
}

type imageURLField struct {
	URL string `json:"url"`
}

type ccContentPart struct {
	Type     string         `json:"type"`
	Text     string         `json:"text,omitempty"`
	ImageURL *imageURLField `json:"image_url,omitempty"`
}

type ccMessage struct {
	Role             string              `json:"role"`
	Content          any                 `json:"content"` // string, []ccContentPart, or nil
	ToolCalls        []shared.CCToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string              `json:"tool_call_id,omitempty"`
	ReasoningContent string              `json:"reasoning_content,omitempty"`
}

type ccRequest struct {
	Model             string          `json:"model"`
	Messages          []ccMessage     `json:"messages"`
	MaxTokens         *int64          `json:"max_tokens,omitempty"`
	Stop              any             `json:"stop,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	ReasoningEffort   string          `json:"reasoning_effort,omitempty"`
	Tools             []shared.CCTool `json:"tools,omitempty"`
	ToolChoice        json.RawMessage `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	ResponseFormat    json.RawMessage `json:"response_format,omitempty"`
	StreamOptions     json.RawMessage `json:"stream_options,omitempty"`
}

// encode finalizes and encodes a translated request; the composed types
// are always marshallable.
func encode(req *ccRequest) []byte {
	if req.Messages == nil {
		req.Messages = []ccMessage{}
	}
	b, _ := json.Marshal(req)
	return b
}

// ---- claude (Anthropic Messages) -> Chat Completions (FR-005, AC §B) ----

type claudeBlock = map[string]any

// claudeToChat translates an Anthropic Messages request into a Chat
// Completions request (FR-005): top-level system becomes a system
// message, text blocks become string content, tool_use blocks become
// assistant tool_calls, tool_result blocks become role:"tool" messages,
// max_tokens/stop_sequences/tools map to their CC equivalents
// (max_tokens defaulted by the shared kernel, FR-005), and the thinking
// budget maps to a capability-aware reasoning_effort via
// thinking.EffortFromBudget. Decoding is owned entirely by the shared
// Claude-request kernel; only target-shape rendering stays local.
func claudeToChat(upstreamModel string, body []byte, ts *pluginapi.ThinkingSupport) ([]byte, *errclass.Error) {
	src, eErr := shared.DecodeClaudeMessages(body)
	if eErr != nil {
		return nil, eErr
	}
	out := &ccRequest{
		Model:       upstreamModel,
		Stream:      src.Stream,
		Temperature: src.Temperature,
		TopP:        src.TopP,
	}
	out.MaxTokens = &src.MaxTokens
	if src.Stream {
		// Usage capture parity with responsesToChat: Ollama only
		// emits the usage trailer when include_usage is requested, and
		// CPA derives manager usage/TPS from the translated stream.
		out.StreamOptions = json.RawMessage(`{"include_usage":true}`)
	}
	if len(src.StopSequences) > 0 {
		out.Stop = src.StopSequences
	}
	if shared.ThinkingEnabled(src.Thinking) {
		out.ReasoningEffort = thinking.EffortFromBudget(src.Thinking.BudgetTokens, ts)
	}
	applyToolChoiceCC(out, src.ToolChoiceKind, src.ToolChoiceName)
	if src.System != "" {
		out.Messages = append(out.Messages, ccMessage{Role: "system", Content: src.System})
	}
	for i := range src.Messages {
		m := &src.Messages[i]
		switch m.Role {
		case "user":
			msgs, eErr := claudeUserMessages(m)
			if eErr != nil {
				return nil, eErr
			}
			out.Messages = append(out.Messages, msgs...)
		case "assistant":
			msg, eErr := claudeAssistantMessage(m)
			if eErr != nil {
				return nil, eErr
			}
			if msg != nil {
				out.Messages = append(out.Messages, *msg)
			}
		default:
			return nil, shared.ValidateRole(m.Role, EndpointPath)
		}
	}
	for _, t := range src.Tools {
		// Claude tools carry no type field; validation cannot reject here.
		schema := shared.ObjectSchema(t.InputSchema)
		out.Tools = append(out.Tools, shared.CCTool{
			Type:     "function",
			Function: shared.CCFunction{Name: t.Name, Description: t.Description, Parameters: schema},
		})
	}
	return encode(out), nil
}

// claudeResultText flattens tool_result content (string or text blocks)
// into a tool-message string via the shared kernel; non-text blocks have
// no tool-message equivalent and are rejected descriptively rather than
// dropped (FR-005).
func claudeResultText(content json.RawMessage) (string, *errclass.Error) {
	return shared.ToolResultText(content, "tool messages carry text only")
}

// claudeUserMessages converts a user turn into Chat Completions messages:
// text/image blocks become one user message (plain string when only
// text), each tool_result block becomes a separate role:"tool" message
// carrying its tool_use_id (FR-005).
func claudeUserMessages(m *shared.ClaudeMessageRecord) ([]ccMessage, *errclass.Error) {
	var msgs []ccMessage
	var parts []ccContentPart
	flush := func() {
		if len(parts) == 0 {
			return
		}
		var c any = parts
		if len(parts) == 1 && parts[0].Type == "text" {
			c = parts[0].Text
		}
		msgs = append(msgs, ccMessage{Role: "user", Content: c})
		parts = nil
	}
	if m.Content != "" {
		parts = append(parts, ccContentPart{Type: "text", Text: m.Content})
	}
	for i := range m.Blocks {
		blk := &m.Blocks[i]
		switch blk.Kind {
		case "text":
			parts = append(parts, ccContentPart{Type: "text", Text: blk.Text})
		case "image":
			parts = append(parts, ccContentPart{Type: "image_url", ImageURL: &imageURLField{URL: blk.URL}})
		case "tool_result":
			text, eErr := claudeResultText(blk.Result)
			if eErr != nil {
				return nil, eErr
			}
			if blk.IsError {
				// Same convention as the Responses-route twin so the
				// flag is never silently erased (FR-005).
				text = shared.ToolResultErrorPrefix + text
			}
			flush()
			msgs = append(msgs, ccMessage{
				Role: "tool", Content: text, ToolCallID: blk.CallID,
			})
		default:
			return nil, shared.UnsupportedPartType(blk.Kind, EndpointPath)
		}
	}
	flush()
	return msgs, nil
}

// claudeAssistantMessage converts an assistant turn: text blocks join
// the message content, tool_use blocks become tool_calls. Historical
// thinking and redacted_thinking blocks have no Chat Completions
// representation and are omitted by the FR-005 explicit omission policy
// (the forward-looking control maps via effortFromBudget instead; the
// redacted variant is pure encrypted metadata). Returns nil for empty turns.
func claudeAssistantMessage(m *shared.ClaudeMessageRecord) (*ccMessage, *errclass.Error) {
	msg := &ccMessage{Role: "assistant"}
	var sb strings.Builder
	sb.WriteString(m.Content)
	for i := range m.Blocks {
		blk := &m.Blocks[i]
		switch blk.Kind {
		case "text":
			sb.WriteString(blk.Text)
		case "thinking", "redacted_thinking":
			// omitted: no Chat Completions equivalent (FR-005 policy)
		case "tool_use":
			tc := shared.CCToolCall{ID: blk.CallID, Type: "function"}
			tc.Function.Name = blk.Name
			tc.Function.Arguments = shared.DefaultArgs(string(blk.Input))
			msg.ToolCalls = append(msg.ToolCalls, tc)
		default:
			return nil, shared.UnsupportedPartType(blk.Kind, EndpointPath)
		}
	}
	if sb.Len() > 0 {
		msg.Content = sb.String()
	}
	if msg.Content == nil && len(msg.ToolCalls) == 0 {
		return nil, nil
	}
	return msg, nil
}

// ---- openai-response (Responses) -> Chat Completions (FR-005) ----

// responsesToChat translates an OpenAI Responses request into a Chat
// Completions request (FR-005): instructions and system items become
// system messages, message items map by role, function_call items merge
// into assistant tool_calls, function_call_output items become role:"tool"
// messages, reasoning.effort maps to reasoning_effort,
// parallel_tool_calls passes through as-is (the reverse leg forwards the
// same field), and max_output_tokens maps to max_tokens. Historical
// reasoning items are omitted (no CC equivalent; FR-005 explicit omission
// policy).
func responsesToChat(upstreamModel string, body []byte, ts *pluginapi.ThinkingSupport, compatibility string) ([]byte, *errclass.Error) {
	var src shared.ResponsesRequest
	if err := json.Unmarshal(body, &src); err != nil {
		return nil, errclass.Translation("malformed openai-response request JSON: " + err.Error())
	}
	if compatibility == "strict" {
		if eErr := shared.ValidateResponsesChatOptions(body); eErr != nil {
			return nil, eErr
		}
	}
	var toolContext shared.ResponsesToolContext
	var chatTools []shared.CCTool
	var eErr *errclass.Error
	if compatibility == "strict" {
		toolContext, chatTools, eErr = shared.BuildResponsesToolContext(&src, EndpointPath)
	} else {
		toolContext, chatTools, eErr = shared.BuildResponsesToolContextCPA(&src, EndpointPath)
	}
	if eErr != nil {
		return nil, eErr
	}
	hadToolDeclarations := len(src.Tools) > 0 || hasAdditionalToolDeclarations(src.Input)
	out := &ccRequest{
		Model:             upstreamModel,
		Stream:            src.Stream,
		Temperature:       src.Temperature,
		TopP:              src.TopP,
		ParallelToolCalls: src.ParallelToolCalls,
		Tools:             chatTools,
	}
	if src.Text != nil && shared.HasContent(src.Text.Format) {
		out.ResponseFormat, eErr = shared.ResponsesTextFormatToChat(src.Text.Format)
		if eErr != nil {
			return nil, eErr
		}
	}
	if src.Stream {
		out.StreamOptions, eErr = responsesStreamOptions(src.StreamOptions)
		if eErr != nil {
			return nil, eErr
		}
	}
	if src.MaxOutputTokens != nil && *src.MaxOutputTokens > 0 {
		out.MaxTokens = src.MaxOutputTokens
	}
	if src.Reasoning != nil && src.Reasoning.Effort != "" {
		if eErr := thinking.ValidateEffort(src.Reasoning.Effort, ts); eErr != nil {
			return nil, eErr
		}
		out.ReasoningEffort = strings.ToLower(strings.TrimSpace(src.Reasoning.Effort))
	}
	kind, tcName, tcNamespace, eErr := shared.DecodeToolChoiceIdentity(src.ToolChoice)
	preserveRawToolChoice := false
	if eErr != nil {
		if compatibility == "strict" {
			return nil, eErr
		}
		kind = shared.ToolChoiceAbsent
		preserveRawToolChoice = len(chatTools) > 0 && shared.HasContent(src.ToolChoice)
	}
	if compatibility != "strict" && hadToolDeclarations && len(chatTools) == 0 {
		kind = shared.ToolChoiceAbsent
	}
	if kind == shared.ToolChoiceNamed {
		identity, resolveErr := toolContext.ResolveCall(tcNamespace, tcName)
		if resolveErr != nil {
			if len(chatTools) != 0 || tcNamespace != "" {
				return nil, resolveErr
			}
		} else {
			tcName = identity.ChatName
		}
	}
	if preserveRawToolChoice {
		out.ToolChoice = src.ToolChoice
	} else {
		applyToolChoiceCC(out, kind, tcName)
	}

	addSystem := func(text string) {
		if text != "" {
			out.Messages = append(out.Messages, ccMessage{Role: "system", Content: text})
		}
	}
	instr, eErr := src.DecodeInstructions()
	if eErr != nil {
		return nil, eErr
	}
	addSystem(instr)

	items, eErr := src.DecodeInputItems()
	if eErr != nil {
		return nil, eErr
	}
	for _, item := range items {
		kind := shared.NormalizeResponsesItemType(item)
		if kind == "" {
			continue
		}
		switch kind {
		case "message":
			content, eErr := respContent(item.Content)
			if eErr != nil {
				return nil, eErr
			}
			switch item.Role {
			case "system", "developer":
				// Flatten to text; images have no system-message
				// equivalent and are rejected descriptively (FR-005).
				text := ""
				if s, ok := content.(string); ok {
					text = s
				} else if parts, ok := content.([]ccContentPart); ok {
					var sb strings.Builder
					for _, p := range parts {
						if p.Type != "text" {
							return nil, shared.SystemImageRejected()
						}
						sb.WriteString(p.Text)
					}
					text = sb.String()
				}
				addSystem(text)
			case "user", "assistant":
				if content != nil {
					out.Messages = append(out.Messages, ccMessage{Role: item.Role, Content: content})
				}
			default:
				return nil, shared.ValidateRole(item.Role, EndpointPath)
			}
		case "function_call":
			chatName, resolveErr := responsesCallChatName(toolContext, len(chatTools), item.Namespace, item.Name)
			if resolveErr != nil {
				return nil, resolveErr
			}
			tc := shared.CCToolCall{ID: item.CallID, Type: "function"}
			tc.Function.Name = chatName
			tc.Function.Arguments = shared.DefaultArgs(item.Arguments)
			// Merge consecutive function_call items into one
			// assistant message so multi-call turns round-trip.
			if n := len(out.Messages); n > 0 {
				last := &out.Messages[n-1]
				if last.Role == "assistant" && last.Content == nil {
					last.ToolCalls = append(last.ToolCalls, tc)
					continue
				}
			}
			out.Messages = append(out.Messages, ccMessage{
				Role: "assistant", ToolCalls: []shared.CCToolCall{tc},
			})
		case "custom_tool_call":
			identity, resolveErr := toolContext.ResolveCall(item.Namespace, item.Name)
			if resolveErr != nil {
				return nil, resolveErr
			}
			if identity.Kind != "custom" {
				return nil, &errclass.Error{Class: errclass.ClassUnsupported, StatusCode: 400, Message: "custom tool call does not match a custom declaration"}
			}
			tc := shared.CCToolCall{ID: item.CallID, Type: "function"}
			tc.Function.Name = identity.ChatName
			tc.Function.Arguments = shared.WrapCustomToolInput(item.Input)
			if n := len(out.Messages); n > 0 {
				last := &out.Messages[n-1]
				if last.Role == "assistant" && last.Content == nil {
					last.ToolCalls = append(last.ToolCalls, tc)
					continue
				}
			}
			out.Messages = append(out.Messages, ccMessage{Role: "assistant", ToolCalls: []shared.CCToolCall{tc}})
		case "function_call_output", "custom_tool_call_output":
			text, eErr := shared.FunctionCallOutputText(item.Output)
			if eErr != nil {
				return nil, eErr
			}
			out.Messages = append(out.Messages, ccMessage{
				Role: "tool", Content: text, ToolCallID: item.CallID,
			})
		case "reasoning":
			// Historical reasoning items remain omitted unless a future lane
			// implements CPA's full session reasoning reconstruction.
		case "additional_tools":
			// Declaration carrier consumed by BuildResponsesToolContext; it is
			// not conversational input and must not become a message.
		default:
			if compatibility != "strict" && isCPAIgnoredResponsesItem(item.Type) {
				continue
			}
			return nil, shared.UnsupportedInputItemType(item.Type)
		}
	}

	return encode(out), nil
}

func hasAdditionalToolDeclarations(input json.RawMessage) bool {
	if !shared.HasContent(input) {
		return false
	}
	var items []struct {
		Type  string            `json:"type"`
		Tools []json.RawMessage `json:"tools"`
	}
	if json.Unmarshal(input, &items) != nil {
		return false
	}
	for _, item := range items {
		if item.Type == "additional_tools" && len(item.Tools) > 0 {
			return true
		}
	}
	return false
}

func isCPAIgnoredResponsesItem(kind string) bool {
	switch kind {
	case "code_interpreter_call", "web_search_call", "file_search_call", "computer_call", "computer_call_output":
		return true
	}
	return false
}

func responsesCallChatName(ctx shared.ResponsesToolContext, declarationCount int, namespace, name string) (string, *errclass.Error) {
	identity, eErr := ctx.ResolveCall(namespace, name)
	if eErr == nil {
		return identity.ChatName, nil
	}
	if namespace == "" && !strings.Contains(eErr.Message, "ambiguous") {
		return name, nil // historical flat function call without declarations
	}
	return "", eErr
}

func responsesStreamOptions(raw json.RawMessage) (json.RawMessage, *errclass.Error) {
	if !shared.HasContent(raw) {
		return json.RawMessage(`{"include_usage":true}`), nil
	}
	var options map[string]json.RawMessage
	if err := json.Unmarshal(raw, &options); err != nil || options == nil {
		return nil, errclass.Translation("stream_options must be an object or null")
	}
	include := true
	if value, ok := options["include_usage"]; ok {
		if err := json.Unmarshal(value, &include); err != nil {
			return nil, errclass.Translation("stream_options.include_usage must be boolean")
		}
	}
	out, _ := json.Marshal(map[string]bool{"include_usage": include})
	return out, nil
}

// respContent normalizes a Responses content field (JSON string or typed
// parts) into Chat Completions content: a plain string, content parts,
// or nil when empty (FR-005 multimodal preservation). The shared kernel
// owns validation and the empty-input policy — empty-text messages drop
// identically on every route (F5 parity).
func respContent(raw json.RawMessage) (any, *errclass.Error) {
	parts, eErr := shared.DecodeStringOrParts(raw, EndpointPath)
	if eErr != nil {
		return nil, eErr
	}
	switch {
	case len(parts) == 0:
		return nil, nil
	case len(parts) == 1 && parts[0].ImageURL == "":
		return parts[0].Text, nil
	}
	out := make([]ccContentPart, 0, len(parts))
	for _, p := range parts {
		if p.ImageURL != "" {
			out = append(out, ccContentPart{Type: "image_url", ImageURL: &imageURLField{URL: p.ImageURL}})
			continue
		}
		out = append(out, ccContentPart{Type: "text", Text: p.Text})
	}
	return out, nil
}

// applyToolChoiceCC maps a normalized tool choice onto the outbound Chat
// Completions request, which natively represents every normalized kind
// (FR-005). It is infallible: shared.DecodeToolChoice emits exactly the
// closed set below, so the switch covers every possible kind (the
// exhaustiveness test in request_test.go enforces this against
// shared.AllToolChoiceKinds); absent omits the field.
func applyToolChoiceCC(req *ccRequest, kind, name string) {
	switch kind {
	case shared.ToolChoiceAuto:
		req.ToolChoice = json.RawMessage(`"auto"`)
	case shared.ToolChoiceAny:
		req.ToolChoice = json.RawMessage(`"required"`)
	case shared.ToolChoiceNone:
		req.ToolChoice = json.RawMessage(`"none"`)
	case shared.ToolChoiceNamed:
		b, _ := json.Marshal(map[string]any{
			"type": "function", "function": map[string]any{"name": name},
		})
		req.ToolChoice = b
	}
}
