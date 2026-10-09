package messages

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"ollama-cliproxyapi/internal/adapter/shared"
	"ollama-cliproxyapi/internal/errclass"
)

// StreamConverter converts an upstream Anthropic Messages SSE stream
// (typed `event:`/`data:` pairs) into the client protocol incrementally
// (FR-006, AC §C streaming). It is stateful and sequential-use only:
// one instance per upstream response, Feed called with consecutive
// network chunks in order.
type StreamConverter struct {
	framer           *shared.SSEFramer
	sourceFormat     string
	created          int64
	blocks           map[int]*blockState
	toolOrdinal      int
	msgID            string
	model            string
	promptTokens     int64
	completionTokens int64
	cacheRead        *int64
	cacheCreation    *int64
	stopReason       string // last observed message_delta stop_reason (responses terminal status)
	toolsSeen        bool   // any tool_use block observed (openai finish-reason precedence)
	roleSent         bool
	msgIdx           int  // responses: compacted output_index of the ONE announced message item (-1 until then)
	outCount         int  // responses: next compacted output position; thinking blocks consume none (FR-005 omission)
	emitted          bool // any client event emitted (Flush eligibility)
	flushed          bool // one-shot guard for Flush
	streamState      shared.StreamState
	lastUsage        usageCounts
	seenStarts       map[int]string
}

func (sc *StreamConverter) StreamSnapshot() shared.StreamSnapshot { return sc.streamState.Snapshot }

// blockState tracks one open upstream content block by index.
type blockState struct {
	kind    string
	id      string
	name    string
	ordinal int             // compact tool_calls index for openai chunks
	outIdx  int             // responses: compacted output_index assigned at block start
	emitted bool            // openai: id/name attached to the first arguments fragment
	text    strings.Builder // accumulated text_delta content (responses target)
	args    strings.Builder // accumulated input_json_delta content (responses target)
}

// NewStreamConverter prepares stream conversion for sourceFormat
// ("openai" Chat Completions chunks, "openai-response" Responses events,
// "claude" verbatim passthrough).
func NewStreamConverter(sourceFormat string) *StreamConverter {
	return &StreamConverter{
		framer:       shared.NewSSEFramer(sourceFormat == "claude"),
		sourceFormat: sourceFormat,
		created:      time.Now().Unix(),
		blocks:       map[int]*blockState{},
		seenStarts:   map[int]string{},
		msgIdx:       -1,
	}
}

// Feed consumes one network chunk and returns fully framed client events,
// whether the upstream stream is finished (message_stop seen), and a
// classified error. Partial lines are buffered until complete.
func (sc *StreamConverter) Feed(chunk []byte) (events [][]byte, done bool, eErr *errclass.Error) {
	sc.framer.Push(chunk)
	for {
		etype, data, rawBlock, ok := sc.framer.Next()
		if !ok {
			return events, false, nil
		}
		d, eErr := sc.dispatch(etype, data, rawBlock, &events)
		if len(events) > 0 {
			sc.emitted = true
		}
		if eErr != nil {
			return events, false, eErr
		}
		if d {
			if _, vErr := sc.validateTools(); vErr != nil {
				return events, false, vErr
			}
		}
		sc.observeFrame(etype, data)
		if d {
			return events, true, nil
		}
	}
}

func (sc *StreamConverter) observeFrame(etype, data string) {
	var ev sseEvent
	_ = json.Unmarshal([]byte(data), &ev)
	switch etype {
	case "message_start":
		if !sc.streamState.Snapshot.Started {
			sc.streamState.Start()
		}
	case "content_block_start":
		idx := int(ev.Index)
		sig := ev.ContentBlock.Type + "\x00" + ev.ContentBlock.ID + "\x00" + ev.ContentBlock.Name
		if sc.seenStarts[idx] != sig {
			sc.seenStarts[idx] = sig
			sc.streamState.Advance()
		}
	case "content_block_delta":
		if ev.Delta.Text != "" || ev.Delta.PartialJSON != "" {
			sc.streamState.Advance()
		}
	case "message_delta":
		if strings.TrimSpace(ev.Delta.StopReason) != "" && !sc.streamState.Snapshot.FinishSeen {
			sc.streamState.Finish()
		}
		if !sameUsage(ev.Usage, sc.lastUsage) {
			sc.lastUsage = ev.Usage
			sc.streamState.Advance()
		}
	case "message_stop":
		if _, e := sc.validateTools(); e == nil {
			sc.streamState.Done()
			sc.streamState.Terminal("message_stop")
		}
	}
}

func (sc *StreamConverter) validateTools() (bool, *errclass.Error) {
	for _, bs := range sc.blocks {
		if bs.kind == "tool_use" {
			raw := shared.DefaultArgs(bs.args.String())
			var v any
			if json.Unmarshal([]byte(raw), &v) != nil {
				return false, errclass.Translation("Messages tool arguments were incomplete at stream completion")
			}
		}
	}
	return true, nil
}

func sameUsage(a, b usageCounts) bool {
	return a.InputTokens == b.InputTokens && a.OutputTokens == b.OutputTokens && floatPtrEqual(a.CacheRead, b.CacheRead) && floatPtrEqual(a.CacheCreation, b.CacheCreation)
}
func floatPtrEqual(a, b *float64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// dispatch routes one complete SSE frame. Conversion targets parse the
// data payload eagerly into typed per-event structs (with the
// type-from-payload fallback when the event: line was absent); claude
// passthrough forwards the verbatim raw block and parses only what error
// handling and done-detection require.
func (sc *StreamConverter) dispatch(etype, data string, rawBlock []byte, events *[][]byte) (bool, *errclass.Error) {
	if sc.sourceFormat == "claude" {
		return sc.dispatchClaude(etype, data, rawBlock, events)
	}
	var ev sseEvent
	if data != "" {
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return false, errclass.Translation(fmt.Sprintf("malformed SSE data JSON: %s", shared.RedactedSnippet(data)))
		}
		if etype == "" {
			etype = ev.Type // tolerate missing event: line
		}
	}
	switch sc.sourceFormat {
	case "openai":
		return sc.dispatchOpenAI(etype, &ev, events)
	case "openai-response":
		return sc.dispatchResponses(etype, &ev, events)
	default:
		return false, shared.UnsupportedFormat(sc.sourceFormat, EndpointPath)
	}
}

// sseEvent is the typed union of Anthropic Messages SSE payload shapes on
// conversion targets. One struct serves every event type because the
// type may arrive only inside the payload (missing event: line), so the
// decode must precede routing; absent members stay zero-valued at no
// cost, and unknown keys are ignored. Error stays a raw map because the
// classifier reads an open-ended upstream object.
type sseEvent struct {
	Type         string         `json:"type"`
	Index        float64        `json:"index"`
	Message      startMessage   `json:"message"`
	ContentBlock contentBlock   `json:"content_block"`
	Delta        deltaBody      `json:"delta"`
	Usage        usageCounts    `json:"usage"`
	Error        map[string]any `json:"error"`
}

type startMessage struct {
	ID    string      `json:"id"`
	Model string      `json:"model"`
	Usage usageCounts `json:"usage"`
}

type contentBlock struct {
	Type string `json:"type"`
	ID   string `json:"id"`
	Name string `json:"name"`
}

type deltaBody struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	PartialJSON string `json:"partial_json"`
	StopReason  string `json:"stop_reason"`
}

type usageCounts struct {
	InputTokens   float64  `json:"input_tokens"`
	OutputTokens  float64  `json:"output_tokens"`
	CacheRead     *float64 `json:"cache_read_input_tokens"`
	CacheCreation *float64 `json:"cache_creation_input_tokens"`
}

// dispatchClaude passes event blocks through verbatim; an in-stream error
// event becomes a classified error instead of reaching the client raw.
// The payload is parsed lazily: only when the typed fallback must resolve
// a missing event: line or when the frame is an error event.
func (sc *StreamConverter) dispatchClaude(etype, data string, raw []byte, events *[][]byte) (bool, *errclass.Error) {
	var payload map[string]any
	if etype == "" && data != "" {
		json.Unmarshal([]byte(data), &payload) // best-effort type probe
		etype, _ = payload["type"].(string)
	}
	if etype == "error" {
		if payload == nil && data != "" {
			json.Unmarshal([]byte(data), &payload)
		}
		return false, sseError(mget(payload, "error"))
	}
	if len(raw) > 0 {
		*events = append(*events, raw)
	}
	return etype == "message_stop", nil
}

// dispatchOpenAI synthesizes chat.completion.chunk frames (FR-006).
func (sc *StreamConverter) dispatchOpenAI(etype string, ev *sseEvent, events *[][]byte) (bool, *errclass.Error) {
	switch etype {
	case "message_start":
		sc.msgID = ev.Message.ID
		sc.model = ev.Message.Model
		sc.promptTokens = int64(ev.Message.Usage.InputTokens)
		sc.captureCache(ev.Message.Usage)
		sc.emitRole(events)
	case "content_block_start":
		bs := &blockState{kind: ev.ContentBlock.Type, id: ev.ContentBlock.ID, name: ev.ContentBlock.Name}
		if bs.kind == "tool_use" {
			bs.ordinal = sc.toolOrdinal
			sc.toolOrdinal++
			sc.toolsSeen = true
		}
		sc.blocks[int(ev.Index)] = bs
	case "content_block_delta":
		bs := sc.blocks[int(ev.Index)]
		switch ev.Delta.Type {
		case "text_delta":
			sc.emitRole(events)
			sc.emitChunk(events, map[string]any{"content": ev.Delta.Text}, "", false)
		case "input_json_delta":
			if bs == nil || bs.kind != "tool_use" {
				return false, errclass.Translation("tool arguments fragment without a tool_use block start")
			}
			var tc map[string]any
			if !bs.emitted {
				tc = shared.CCToolCallOpeningEntry(bs.ordinal, bs.id, bs.name, ev.Delta.PartialJSON)
				bs.emitted = true
			} else {
				tc = map[string]any{
					"index":    bs.ordinal,
					"function": map[string]any{"arguments": ev.Delta.PartialJSON},
				}
			}
			sc.emitRole(events)
			sc.emitChunk(events, map[string]any{"tool_calls": []any{tc}}, "", false)
		case "thinking_delta":
			// FR-005 explicit omission policy: chat.completion.chunk has no
			// standard reasoning-delta field, so thinking text is dropped
			// rather than put in a non-standard field. Signatures have no
			// destination either.
		}
	case "message_delta":
		if ev.Usage.InputTokens != 0 {
			sc.promptTokens = int64(ev.Usage.InputTokens)
		}
		sc.completionTokens = int64(ev.Usage.OutputTokens)
		sc.captureCache(ev.Usage)
		// Observed tool calls outrank the status-derived reason: a
		// terminal max_tokens cannot downgrade tool_calls to length.
		sc.emitChunk(events, map[string]any{},
			shared.TerminalReason(sc.toolsSeen, "tool_calls",
				shared.ClaudeStopToFinish(ev.Delta.StopReason)), true)
	case "message_stop":
		return true, nil
	case "error":
		return false, sseError(ev.Error)
	default: // ping and unknown event types are ignored (forward compat)
	}
	return false, nil
}

// dispatchResponses synthesizes Responses SSE events (FR-006).
func (sc *StreamConverter) dispatchResponses(etype string, ev *sseEvent, events *[][]byte) (bool, *errclass.Error) {
	switch etype {
	case "message_start":
		sc.msgID = ev.Message.ID
		sc.model = ev.Message.Model
		sc.promptTokens = int64(ev.Message.Usage.InputTokens)
		sc.captureCache(ev.Message.Usage)
		// Lifecycle parity with the Chat Completions route's synthesizer:
		// announce the response before any deltas reference it (F18).
		*events = append(*events, sc.responsesEm().Created())
	case "content_block_start":
		bs := &blockState{kind: ev.ContentBlock.Type, id: ev.ContentBlock.ID, name: ev.ContentBlock.Name}
		sc.blocks[int(ev.Index)] = bs
		// Announce each translatable item before its first delta (F18
		// lifecycle parity); thinking blocks produce no deltas and are
		// omitted entirely (FR-005). Indexes come from the SAME compacted
		// space outputItems() renders, so streamed output_index values
		// always match terminal response.completed.output positions.
		switch bs.kind {
		case "text":
			if sc.msgIdx >= 0 {
				break // one message item aggregates every text block (canonical shape)
			}
			sc.msgIdx = sc.outCount
			sc.outCount++
			// Native-compatible shape: the announced item carries "id" so
			// downstream output_text.delta item_ids resolve (F-R3 parity).
			*events = append(*events, sc.responsesEm().ItemAdded(sc.msgIdx, map[string]any{
				"type": "message", "id": sc.msgID, "role": "assistant", "content": []any{},
			}))
		case "tool_use":
			sc.toolsSeen = true
			bs.outIdx = sc.outCount
			sc.outCount++
			*events = append(*events, sc.responsesEm().ItemAdded(bs.outIdx, map[string]any{
				"type": "function_call", "call_id": bs.id, "name": bs.name, "arguments": "",
			}))
		}
	case "content_block_delta":
		idx := int(ev.Index)
		bs := sc.blocks[idx]
		switch ev.Delta.Type {
		case "text_delta":
			if bs != nil {
				bs.text.WriteString(ev.Delta.Text)
			}
			// Deltas target the single aggregated message item, so later
			// text blocks reference its announced position, never their own.
			dIdx := idx
			if sc.msgIdx >= 0 {
				dIdx = sc.msgIdx
			}
			*events = append(*events, sc.responsesEm().TextDelta(sc.msgID, dIdx, ev.Delta.Text))
		case "input_json_delta":
			if bs == nil || bs.kind != "tool_use" {
				return false, errclass.Translation("tool arguments fragment without a tool_use block start")
			}
			bs.args.WriteString(ev.Delta.PartialJSON)
			*events = append(*events, sc.responsesEm().ArgsDelta(bs.id, bs.outIdx, ev.Delta.PartialJSON))
		case "thinking_delta":
			// FR-005 explicit omission policy: no standard Responses
			// reasoning-delta event; dropped.
		}
	case "message_delta":
		if ev.Usage.InputTokens != 0 {
			sc.promptTokens = int64(ev.Usage.InputTokens)
		}
		sc.completionTokens = int64(ev.Usage.OutputTokens) // accumulated, reported on completion
		sc.captureCache(ev.Usage)
		sc.stopReason = ev.Delta.StopReason
	case "message_stop":
		*events = append(*events, sc.responsesCompleted())
		return true, nil
	case "error":
		return false, sseError(ev.Error)
	default: // ping and unknown event types are ignored (forward compat)
	}
	return false, nil
}

// responsesCompleted builds the terminal response.completed event: shape
// parity with the non-stream claudeToResponses converter — status comes
// from the Responses vocabulary {completed, incomplete} via the observed
// stop_reason (max_tokens truncation → incomplete); tool calls are
// represented by their function_call output items and never override the
// status (FR-006). Shared by message_stop and Flush so an early upstream
// close cannot diverge from the normal-path shape.
func (sc *StreamConverter) responsesCompleted() []byte {
	status := shared.ResponseStatusFromClaudeStop(sc.stopReason)
	usage := shared.NewResponsesUsageFrom(sc.promptTokens+valueOrZero(sc.cacheRead)+valueOrZero(sc.cacheCreation), sc.completionTokens,
		shared.UsageDetails{CachedTokens: sc.cacheRead, CacheWriteTokens: sc.cacheCreation})
	return sc.responsesEm().Completed(status, usage, sc.outputItems())
}

// responsesEm binds the shared Responses emitter kernel to the captured
// upstream message identity so this route's frames cannot diverge from the
// sibling Chat-Completions-route synthesizer (FR-006).
func (sc *StreamConverter) responsesEm() shared.ResponsesEventEmitter {
	return shared.ResponsesEventEmitter{ID: sc.msgID, Model: sc.model}
}

// Flush terminates a stream whose upstream closed before message_stop:
// without this the client never sees response.completed nor [DONE] (F5,
// FR-006). Responses synthesis emits response.completed from captured
// identity, usage, and accumulated output items; openai targets get the
// missing [DONE]; claude passthrough holds nothing back. One-shot: later
// calls return nothing, as does any call with nothing emitted.
func (sc *StreamConverter) Flush() [][]byte {
	if sc.flushed {
		return nil
	}
	sc.flushed = true
	if !sc.emitted {
		return nil
	}
	switch sc.sourceFormat {
	case "openai-response":
		return [][]byte{sc.responsesCompleted()}
	case "openai":
		return nil
	default: // claude passthrough forwards verbatim; nothing deferred
		return nil
	}
}

func (sc *StreamConverter) FinalizeStream() ([][]byte, *errclass.Error) {
	if sc.streamState.Snapshot.TerminalSeen {
		return nil, nil
	}
	if !sc.streamState.Snapshot.FinishSeen {
		return nil, errclass.Translation("Messages stream ended before a terminal stop reason")
	}
	if _, e := sc.validateTools(); e != nil {
		return nil, e
	}
	events := sc.Flush()
	sc.streamState.Terminal("finish")
	return events, nil
}

func (sc *StreamConverter) FlushWithError() ([][]byte, *errclass.Error) { return sc.FinalizeStream() }

// outputItems materializes the Responses output array from observed
// content blocks in upstream index order (FR-006) through the shared
// OutputAssembler: ALL text blocks aggregate into ONE message item
// positioned at the first text block's slot while tool_use blocks become
// separate function_call items — the canonical native shape, equal to the
// non-stream claudeToResponses converter (FR-006 mode parity); thinking
// blocks are omitted (FR-005).
func (sc *StreamConverter) outputItems() []any {
	indexes := make([]int, 0, len(sc.blocks))
	for i := range sc.blocks {
		indexes = append(indexes, i)
	}
	sort.Ints(indexes)
	oa := shared.NewOutputAssembler(sc.msgID)
	for _, i := range indexes {
		bs := sc.blocks[i]
		switch bs.kind {
		case "text":
			oa.ReserveTextSlot()
			oa.AddText(bs.text.String())
		case "tool_use":
			oa.AppendFunctionCall(bs.id, bs.name, shared.DefaultArgs(bs.args.String()))
		}
	}
	return oa.Render()
}

// chatChunks binds the shared chunk kernel to the captured upstream
// message identity so this route's frames cannot diverge from the
// sibling Responses-route synthesizer (FR-006).
func (sc *StreamConverter) chatChunks() shared.ChatChunkBuilder {
	return shared.ChatChunkBuilder{ID: sc.msgID, Model: sc.model, Created: sc.created}
}

// emitRole guarantees the leading assistant-role chunk openai clients expect.
func (sc *StreamConverter) emitRole(events *[][]byte) {
	if sc.roleSent {
		return
	}
	sc.roleSent = true
	*events = append(*events, sc.chatChunks().RoleChunk())
}

// emitChunk appends one synthesized chat.completion.chunk frame via the
// shared kernel; an empty finish renders null (intermediate chunks) and
// withUsage attaches the accumulated token usage to the terminal frame.
func (sc *StreamConverter) emitChunk(events *[][]byte, delta map[string]any, finish string, withUsage bool) {
	b := sc.chatChunks()
	if finish == "" {
		*events = append(*events, b.Delta(delta))
		return
	}
	var usage map[string]any
	if withUsage {
		usage = shared.CCUsageFrom(sc.promptTokens+valueOrZero(sc.cacheRead)+valueOrZero(sc.cacheCreation), sc.completionTokens,
			shared.UsageDetails{CachedTokens: sc.cacheRead})
	}
	*events = append(*events, b.Finish(finish, usage))
}

func (sc *StreamConverter) captureCache(u usageCounts) {
	if u.CacheRead != nil {
		v := int64(*u.CacheRead)
		sc.cacheRead = &v
	}
	if u.CacheCreation != nil {
		v := int64(*u.CacheCreation)
		sc.cacheCreation = &v
	}
}

// sseError maps an Anthropic in-stream error event onto the shared
// classification (FR-009) via its documented error-type taxonomy.
func sseError(e map[string]any) *errclass.Error {
	errorType := sget(e, "type")
	message := sget(e, "message")
	if strings.Contains(strings.ToLower(errorType), "gousagelimiterror") || strings.Contains(strings.ToLower(errorType), "freeusagelimiterror") {
		message = errorType + ": " + message
	}
	return errclass.FromStatus(anthropicErrorStatus(errorType), message)
}

func anthropicErrorStatus(t string) int {
	switch t {
	case "invalid_request_error":
		return 400
	case "authentication_error":
		return 401
	case "permission_error":
		return 403
	case "not_found_error":
		return 404
	case "request_too_large":
		return 413
	case "rate_limit_error":
		return 429
	case "overloaded_error":
		return 503
	default:
		return 500
	}
}

func mget(m map[string]any, k string) map[string]any {
	v, _ := m[k].(map[string]any)
	return v
}

func sget(m map[string]any, k string) string {
	v, _ := m[k].(string)
	return v
}
