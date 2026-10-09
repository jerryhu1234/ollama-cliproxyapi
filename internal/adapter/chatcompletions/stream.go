package chatcompletions

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"

	"ollama-cliproxyapi/internal/adapter/shared"
	"ollama-cliproxyapi/internal/errclass"
)

// StreamConverter incrementally converts an upstream Chat Completions
// SSE stream into the client protocol's stream (FR-006, AC §B): "openai"
// frames pass through verbatim (with reasoning_content backfilled from
// vendor reasoning spellings), "claude" synthesizes Anthropic Messages
// events, "openai-response" synthesizes Responses events. Partial SSE
// lines are buffered across Feed calls; tool-call argument fragments are
// accumulated per tool_calls index until the stream terminates.
//
// A converter is safe for SEQUENTIAL Feed calls only — it is not
// goroutine-safe; use one converter per upstream stream. Synthesis
// follows choices[0] (upstream serves n=1 routes).
type StreamConverter struct {
	sourceFormat string
	lineBuf      []byte // partial SSE line carried across Feed calls
	done         bool

	started       bool // message_start / first chunk seen
	id            string
	model         string
	claudeEm      shared.ClaudeEventEmitter // canonical Messages frames bound to first-chunk identity
	textOpen      bool                      // claude text content_block open
	textIndex     int                       // claude index of the currently-open text block
	thinkOpen     bool                      // claude thinking content_block open
	thinkIndex    int                       // claude index of the currently-open thinking block
	nextIndex     int                       // next output/block index
	msgIndex      int                       // announced assistant message item index (-1 until text)
	thinkItemID   string                    // announced reasoning item identity ("" until reasoning)
	thinkIndexIn  int                       // announced reasoning output index (-1 until reasoning)
	thinkStopped  bool                      // reasoning done transitions already emitted
	thinkSeen     bool                      // any reasoning text observed (terminal item inclusion)
	thinkBuf      strings.Builder           // aggregated reasoning summary text
	tools         map[int64]*streamTool
	toolOrder     []int64 // upstream tool_call indices in first-arrival order
	toolsSeen     bool    // any tool_calls entry observed (terminal-reason precedence)
	usage         *ccUsage
	finished      bool            // finish_reason processed
	heldFinish    string          // finish_reason awaiting terminal emission (flushed on the next data line or [DONE], so the standard include_usage trailer lands in the terminal event; Flush covers close-without-[DONE])
	terminalSent  bool            // claudeTerminal already emitted message_delta (Flush must still close with message_stop)
	flushed       bool            // Flush already ran (one-shot guard)
	respText      strings.Builder // openai-response accumulated output_text
	toolContext   shared.ResponsesToolContext
	respSequence  int64
	msgItemID     string
	textPartOpen  bool
	textClosed    bool
	responseItems map[int]any
	reasonOrdinal int
	streamState   shared.StreamState
	usageProgress string
}

// StreamSnapshot reports the lifecycle counters for the finish log.
func (sc *StreamConverter) StreamSnapshot() shared.StreamSnapshot { return sc.streamState.Snapshot }

// UsageTrace reports whether upstream delivered a usage trailer and its
// final counts, for the stream-finished log (debug-only).
func (sc *StreamConverter) UsageTrace() UsageTrace {
	trace := UsageTrace{}
	if sc.usage != nil {
		trace.Seen = true
		trace.InputTokens = sc.usage.PromptTokens
		trace.OutputTokens = sc.usage.CompletionTokens
	}
	return trace
}

// UsageTrace mirrors the usage captured by the decoder so the stream
// finish log can distinguish "upstream sent no usage" (trailer never
// arrived) from "upstream sent zeros" — the two causes a manager showing
// 0/0 tokens needs telling apart. Debug-only instrumentation; token
// values are metadata, not response content.
type UsageTrace struct {
	Seen         bool
	InputTokens  int64
	OutputTokens int64
}

// streamTool accumulates one upstream tool_calls index; args collects
// argument fragments so the terminal response.completed output carries the
// complete call (FR-006).
type streamTool struct {
	blockIndex   int
	id           string
	name         string
	args         strings.Builder
	stopped      bool // content_block_stop emitted
	announced    bool // Responses output_item.added emitted
	custom       bool // request provenance classified this call as custom
	argsSent     int  // bytes already emitted as function_call_arguments.delta
	identity     shared.ResponsesToolIdentity
	responseDone bool
}

// NewStreamConverter returns a converter translating Chat Completions
// SSE into sourceFormat's stream shape.
func NewStreamConverter(sourceFormat string, originalRequest ...[]byte) *StreamConverter {
	return &StreamConverter{
		sourceFormat:  sourceFormat,
		toolContext:   responseToolContext(originalRequest),
		msgIndex:      -1,
		thinkIndexIn:  -1,
		tools:         map[int64]*streamTool{},
		responseItems: map[int]any{},
	}
}

// Feed consumes one upstream chunk (any split of the byte stream),
// returning the client-protocol events completed by this chunk, whether
// the stream is done ([DONE] seen), and a classified error for
// malformed chunks (FR-006). Errors carry at most an 80-character
// redacted snippet — never the full upstream body.
func (sc *StreamConverter) Feed(chunk []byte) (events [][]byte, done bool, eErr *errclass.Error) {
	if sc.done {
		return nil, true, nil
	}
	sc.lineBuf = append(sc.lineBuf, chunk...)
	for {
		i := bytes.IndexByte(sc.lineBuf, '\n')
		if i < 0 {
			break // keep trailing partial line buffered
		}
		line := strings.TrimSuffix(string(sc.lineBuf[:i]), "\r")
		sc.lineBuf = sc.lineBuf[i+1:]
		ev, eErr := sc.handleLine(line)
		if eErr != nil {
			return events, false, eErr
		}
		events = append(events, ev...)
		if sc.done {
			break
		}
	}
	return events, sc.done, nil
}

func (sc *StreamConverter) handleLine(line string) ([][]byte, *errclass.Error) {
	switch sc.sourceFormat {
	case "openai":
		return sc.passthroughLine(line)
	case "claude":
		return sc.claudeLine(line)
	case "openai-response":
		return sc.responsesLine(line)
	default:
		return nil, shared.UnsupportedFormat(sc.sourceFormat, EndpointPath)
	}
}

// Flush terminates a stream whose upstream closed without [DONE]: the
// finish_reason chunk may legally be the last data line before close, and
// without this the client never sees message_delta/message_stop or
// response.completed — nor usage stranded behind the deferred terminal
// (FR-006). Claude synthesis appends message_stop after the held
// message_delta; Responses synthesis emits response.completed. One-shot:
// later calls return nothing, as does any call with no pending terminal.
func (sc *StreamConverter) Flush() [][]byte {
	if sc.flushed {
		return nil
	}
	sc.flushed = true
	switch sc.sourceFormat {
	case "claude":
		events := sc.claudeTerminal()
		if len(events) == 0 {
			if !sc.terminalSent {
				return nil
			}
			// The post-finish include_usage trailer already delivered the
			// terminal via claudeLine; close-without-[DONE] must still emit
			// message_stop or Anthropic clients hang until timeout.
			return [][]byte{sc.claudeEm.MessageStop()}
		}
		return append(events, sc.claudeEm.MessageStop())
	case "openai-response":
		events, _ := sc.responsesTerminal(true)
		return events
	default:
		return nil
	}
}

// sseData extracts a data-line payload; ok is false for non-data lines
// (comments, event:, retry:, empty lines), which converters ignore.
//
// KNOWN LIMITATION (audited, accepted): this line-oriented parser treats
// each data: line as one complete payload, so a legal multi-data-line SSE
// frame whose lines form a single JSON document decodes as separate
// fragments and fails unmarshal with a spurious translation error. Real
// Chat Completions upstreams emit single-data-line frames, and the shared
// SSEFramer migration was declined because its frame-granular batching
// conflicts with passthroughLine's line-immediate [DONE] semantics.
func sseData(line string) (payload string, ok bool) {
	rest, ok := strings.CutPrefix(line, "data:")
	if !ok {
		return "", false
	}
	return strings.TrimSpace(rest), true
}

// passthroughLine extracts the bare data payload for openai-target emissions;
// [DONE] terminates the stream without emitting a chunk (CPA's WriteDone
// automatically writes the trailing data: [DONE]). Reasoning text arriving
// under a vendor spelling is mirrored onto the standard reasoning_content
// member so clients that only read that member still see thinking; every
// other payload forwards byte-identical.
func (sc *StreamConverter) passthroughLine(line string) ([][]byte, *errclass.Error) {
	data, ok := sseData(line)
	if !ok {
		return nil, nil
	}
	if shared.IsSSEDone(data) {
		if eErr := sc.validateAccumulatedTools(); eErr != nil {
			return nil, eErr
		}
		sc.done = true
		sc.streamState.Done()
		sc.streamState.Terminal("done")
		return nil, nil
	}
	if data == "" {
		return nil, nil
	}
	var observed ccChunk
	if json.Unmarshal([]byte(data), &observed) == nil {
		if !sc.streamState.Snapshot.Started && len(observed.Choices) > 0 {
			sc.streamState.Start()
		}
		if observed.Usage != nil && sc.usageChanged(observed.Usage) {
			sc.streamState.Advance()
		}
		if len(observed.Choices) > 0 {
			c := observed.Choices[0]
			if c.Delta.Content != "" || len(c.Delta.ToolCalls) > 0 {
				sc.streamState.Advance()
			}
			if _, ok := c.Delta.ReasoningText(); ok {
				sc.streamState.Advance()
			}
			if c.FinishReason != "" && !sc.streamState.Snapshot.FinishSeen {
				sc.streamState.Finish()
			}
			for _, tc := range c.Delta.ToolCalls {
				t := sc.tools[tc.Index]
				if t == nil {
					t = &streamTool{id: tc.ID}
					sc.tools[tc.Index] = t
					sc.toolOrder = append(sc.toolOrder, tc.Index)
				}
				if tc.Function.Name != "" {
					t.name += tc.Function.Name
				}
				if tc.Function.Arguments != "" {
					t.args.WriteString(tc.Function.Arguments)
				}
			}
		}
	}
	if fixed := shared.BackfillReasoningContent([]byte(data), "delta"); fixed != nil {
		return [][]byte{fixed}, nil
	}
	return [][]byte{[]byte(data)}, nil
}

func (sc *StreamConverter) usageChanged(usage *ccUsage) bool {
	if usage == nil {
		return false
	}
	raw, _ := json.Marshal(usage)
	next := string(raw)
	if next == sc.usageProgress {
		return false
	}
	sc.usageProgress = next
	return true
}

func (sc *StreamConverter) validateAccumulatedTools() *errclass.Error {
	for _, idx := range sc.toolOrder {
		t := sc.tools[idx]
		if t == nil {
			continue
		}
		raw := shared.DefaultArgs(t.args.String())
		var value any
		if json.Unmarshal([]byte(raw), &value) != nil {
			return errclass.Translation("tool call arguments were incomplete at stream completion")
		}
	}
	return nil
}

// ---- upstream Chat Completions chunk shape ----

type ccToolCallDelta struct {
	Index    int64  `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type ccDelta struct {
	Content   string            `json:"content"`
	ToolCalls []ccToolCallDelta `json:"tool_calls"`
	// Reasoning carries the vendor thinking spellings (reasoning /
	// reasoning_details[].text / reasoning_content); embedded untagged so
	// the shared kernel decodes all three at once.
	shared.ReasoningFields
}

type ccChunkChoice struct {
	Delta        ccDelta `json:"delta"`
	FinishReason string  `json:"finish_reason"`
}

type ccChunk struct {
	ID      string          `json:"id"`
	Model   string          `json:"model"`
	Choices []ccChunkChoice `json:"choices"`
	Usage   *ccUsage        `json:"usage"`
	Error   *ccChunkError   `json:"error"`
}

// ccChunkError is an in-stream upstream error object; choice-less chunks
// carrying one are failures, not empty deltas.
type ccChunkError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

// decodeChunk parses one data payload; malformed JSON yields a
// translation failure carrying only a short redacted snippet (FR-006,
// §5 security: no upstream body echo). A non-empty upstream error object
// fails the conversion with a classified error so the fail path closes
// both streams instead of the chunk being silently dropped as a
// zero-choices delta.
//
// Usage is captured here — before any empty-choices early return in the
// format converters: OpenAI's stream_options.include_usage terminal
// chunk is exactly choice-less-with-usage, and dropping it would lose
// the final counts.
func (sc *StreamConverter) decodeChunk(data string) (*ccChunk, *errclass.Error) {
	var chunk ccChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return nil, errclass.Translation("malformed Chat Completions stream chunk: " + shared.RedactedSnippet(data))
	}
	if chunk.Error != nil {
		return nil, chunkError(chunk.Error)
	}
	if chunk.Usage != nil {
		sc.usage = chunk.Usage
	}
	return &chunk, nil
}

// chunkError classifies an in-stream Chat Completions error object onto
// FR-009 classes, mirroring the sibling adapters' stream-error paths:
// a numeric code (or rate-limit marker) maps through FromStatus so §7
// retryability semantics apply; anything else is a retryable upstream
// server failure.
func chunkError(e *ccChunkError) *errclass.Error {
	status := 0
	switch c := e.Code.(type) {
	case float64:
		status = int(c)
	case string:
		if n, err := strconv.Atoi(c); err == nil {
			status = n
		} else if strings.Contains(c, "rate_limit") {
			status = 429
		}
	}
	if status == 0 && strings.Contains(e.Type, "rate_limit") {
		status = 429
	}
	if status > 0 {
		return errclass.FromStatus(status, e.Message)
	}
	return errclass.UpstreamFallback(e.Message)
}

// ---- Chat Completions -> claude (Anthropic Messages) stream (FR-006) ----

// claudeLine synthesizes Messages events from one chunk: the first
// chunk opens message_start, delta.content opens the text content block
// lazily on first non-empty delta (tool-only streams therefore carry
// tool_use from index 0 like non-stream chatToClaude) and becomes
// text_delta (reopening a new text block if tool_use starts
// closed it), tool_calls argument fragments open tool_use
// blocks and stream input_json_delta, finish_reason closes blocks and
// holds message_delta with the mapped stop reason until the next data
// line or [DONE] (OpenAI's standard include_usage trailer follows the
// finish_reason chunk and must land in the terminal event), and [DONE]
// emits message_stop. Vendor thinking text opens the leading thinking
// block, which every later block start closes — Anthropic's block order is
// thinking, text, tool_use, one block open at a time.
func (sc *StreamConverter) claudeLine(line string) ([][]byte, *errclass.Error) {
	data, ok := sseData(line)
	if !ok || data == "" {
		return nil, nil
	}
	if shared.IsSSEDone(data) {
		if eErr := sc.validateAccumulatedTools(); eErr != nil {
			return nil, eErr
		}
		sc.done = true
		sc.streamState.Done()
		sc.streamState.Terminal("message_stop")
		events := sc.stopBlocks()
		events = append(events, sc.claudeTerminal()...)
		return append(events, sc.claudeEm.MessageStop()), nil
	}
	chunk, eErr := sc.decodeChunk(data)
	if eErr != nil {
		return nil, eErr
	}
	var events [][]byte
	if !sc.streamState.Snapshot.Started && len(chunk.Choices) > 0 {
		sc.streamState.Start()
	}
	if sc.heldFinish != "" {
		// A post-finish chunk (typically the include_usage trailer)
		// flushes the deferred terminal so it carries the final counts.
		events = append(events, sc.claudeTerminal()...)
	}
	if chunk.Usage != nil && sc.usageChanged(chunk.Usage) {
		sc.streamState.Advance()
	}
	if !sc.started && len(chunk.Choices) > 0 {
		sc.started = true
		input := int64(0)
		if chunk.Usage != nil {
			input = chunk.Usage.PromptTokens
		}
		sc.claudeEm = shared.NewClaudeEventEmitter(chunk.ID, chunk.Model)
		events = append(events, sc.claudeEm.MessageStart(input))
	}
	if len(chunk.Choices) == 0 {
		return events, nil
	}
	choice := chunk.Choices[0]
	if choice.Delta.Content != "" || len(choice.Delta.ToolCalls) > 0 {
		sc.streamState.Advance()
	}
	if _, ok := choice.Delta.ReasoningText(); ok {
		sc.streamState.Advance()
	}
	if choice.FinishReason != "" && !sc.streamState.Snapshot.FinishSeen {
		sc.streamState.Finish()
	}
	if thinking, ok := choice.Delta.ReasoningText(); ok {
		// Vendor thinking spellings become the leading thinking block. A
		// reopen (thinking after text or tools) closes every open block
		// first, the same one-open-block rule the text path follows.
		if !sc.thinkOpen {
			events = append(events, sc.stopBlocks()...)
			sc.thinkIndex = sc.nextIndex
			sc.nextIndex++
			sc.thinkOpen = true
			events = append(events, sc.claudeEm.ThinkingBlockStart(sc.thinkIndex))
		}
		events = append(events, sc.claudeEm.ThinkingDelta(sc.thinkIndex, thinking))
	}
	if choice.Delta.Content != "" {
		// The text block opens lazily on first non-empty content so pure
		// tool-call streams carry no phantom empty text block, matching
		// non-stream chatToClaude's block shape. Every content_block_start
		// (this reopen and a first-of-a-new-index tool_use start below)
		// closes ALL currently-open blocks first — open text via stopText
		// and sibling tool_use blocks ascending — mirroring responses'
		// closeOpenBlocks discipline, because Anthropic forbids two
		// concurrently-open blocks and legal Chat Completions interleaving
		// would otherwise hold both open.
		if !sc.textOpen {
			events = append(events, sc.stopBlocks()...)
			sc.textIndex = sc.nextIndex
			sc.nextIndex++
			sc.textOpen = true
			events = append(events, sc.claudeEm.ContentBlockStart(sc.textIndex, "text", map[string]any{"text": ""}))
		}
		events = append(events, sc.claudeEm.ContentBlockDelta(sc.textIndex,
			map[string]any{"type": "text_delta", "text": choice.Delta.Content}))
	}
	for _, tc := range choice.Delta.ToolCalls {
		t := sc.tools[tc.Index]
		if t == nil {
			events = append(events, sc.stopBlocks()...)
			t = &streamTool{blockIndex: sc.nextIndex, id: tc.ID, name: tc.Function.Name}
			sc.nextIndex++
			sc.tools[tc.Index] = t
			sc.toolsSeen = true
			sc.toolOrder = append(sc.toolOrder, tc.Index)
			events = append(events, sc.claudeEm.ContentBlockStart(t.blockIndex, "tool_use",
				map[string]any{"id": t.id, "name": t.name, "input": map[string]any{}}))
		}
		if tc.Function.Arguments != "" {
			t.args.WriteString(tc.Function.Arguments)
			events = append(events, sc.claudeEm.ContentBlockDelta(t.blockIndex,
				map[string]any{"type": "input_json_delta", "partial_json": tc.Function.Arguments}))
		}
	}
	if choice.FinishReason != "" && !sc.finished {
		sc.finished = true
		sc.heldFinish = choice.FinishReason
		events = append(events, sc.stopBlocks()...)
	}
	return events, nil
}

// claudeTerminal renders the held message_delta exactly once, carrying
// the captured usage when any arrived before emission.
func (sc *StreamConverter) claudeTerminal() [][]byte {
	finish := sc.heldFinish
	sc.heldFinish = ""
	if finish == "" {
		return nil
	}
	sc.terminalSent = true
	// Observed tool calls outrank the status-derived reason, mirroring
	// non-stream chatToClaude (FR-006).
	reason := shared.TerminalReason(sc.toolsSeen, "tool_use", shared.FinishToClaudeStop(finish))
	// Always attach: sibling terminals never omit usage; fields zero when
	// upstream reported none (F-R6).
	input, output := int64(0), int64(0)
	var cacheRead *int64
	if sc.usage != nil {
		input = sc.usage.PromptTokens
		output = sc.usage.CompletionTokens
		if sc.usage.PromptDetails != nil {
			cacheRead = sc.usage.PromptDetails.CachedTokens
		}
	}
	return [][]byte{sc.claudeEm.MessageDelta(&reason,
		shared.ClaudeUsage(shared.ClampSubtract(input, cacheRead), output, cacheRead, nil))}
}

// stopThinking closes the open thinking content block, if any.
func (sc *StreamConverter) stopThinking() [][]byte {
	if !sc.thinkOpen {
		return nil
	}
	sc.thinkOpen = false
	return [][]byte{sc.claudeEm.ContentBlockStop(sc.thinkIndex)}
}

// stopText closes the open text content block, if any.
func (sc *StreamConverter) stopText() [][]byte {
	if !sc.textOpen {
		return nil
	}
	sc.textOpen = false
	return [][]byte{sc.claudeEm.ContentBlockStop(sc.textIndex)}
}

// stopBlocks closes the thinking, text, and every open tool_use block
// exactly once in Anthropic block order; called before every
// content_block_start (so blocks never overlap) and again (as a no-op) on
// finish_reason and [DONE].
func (sc *StreamConverter) stopBlocks() [][]byte {
	var events [][]byte
	events = append(events, sc.stopThinking()...)
	events = append(events, sc.stopText()...)
	for _, idx := range sc.toolOrder {
		t := sc.tools[idx]
		if t != nil && !t.stopped {
			t.stopped = true
			events = append(events, sc.claudeEm.ContentBlockStop(t.blockIndex))
		}
	}
	return events
}

// ---- Chat Completions -> openai-response (Responses) stream (FR-006) ----

// responsesLine synthesizes Responses events: delta.content becomes
// response.output_text.delta, tool_calls argument fragments become
// response.function_call_arguments.delta keyed by item_id, and
// finish_reason holds response.completed — emitted at the next data
// line or [DONE] with prompt/completion tokens mapped to input/output
// usage (the standard include_usage trailer follows the finish_reason
// chunk); [DONE] terminates the stream. Vendor thinking text becomes the
// leading reasoning item with its summary events (host-translator parity),
// closed before the first text or function_call item is announced.
func (sc *StreamConverter) responsesLine(line string) ([][]byte, *errclass.Error) {
	data, ok := sseData(line)
	if !ok || data == "" {
		return nil, nil
	}
	if shared.IsSSEDone(data) {
		events := sc.stopReasoning()
		terminal, terminalErr := sc.responsesTerminal(true)
		if terminalErr != nil {
			return events, terminalErr
		}
		sc.done = true
		sc.streamState.Done()
		return append(events, terminal...), nil
	}
	chunk, eErr := sc.decodeChunk(data)
	if eErr != nil {
		return nil, eErr
	}
	var events [][]byte
	changed := false
	if sc.heldFinish != "" {
		// A post-finish chunk (typically the include_usage trailer)
		// flushes the deferred terminal so it carries the final counts.
		terminal, terminalErr := sc.responsesTerminal(false)
		if terminalErr != nil {
			return events, terminalErr
		}
		events = append(events, terminal...)
	}
	if chunk.Usage != nil && sc.usageChanged(chunk.Usage) {
		sc.streamState.Advance()
	}
	if !sc.started && len(chunk.Choices) > 0 {
		sc.started = true
		sc.id = chunk.ID
		sc.model = chunk.Model
		// Lifecycle parity with the Responses protocol: announce the
		// response before any deltas reference it (F18).
		events = append(events, sc.responsesEm().Created(), sc.responsesEm().InProgress())
		sc.streamState.Start()
		changed = true
	}
	if len(chunk.Choices) == 0 {
		return events, nil
	}
	choice := chunk.Choices[0]
	if thinking, ok := choice.Delta.ReasoningText(); ok {
		changed = true
		events = append(events, sc.closeResponseText("completed")...)
		if sc.thinkStopped {
			sc.thinkStopped = false
			sc.thinkSeen = false
			sc.thinkBuf.Reset()
			sc.reasonOrdinal++
		}
		if !sc.thinkSeen {
			sc.thinkSeen = true
			sc.thinkIndexIn = sc.nextIndex
			sc.nextIndex++
			sc.thinkItemID = shared.ReasoningItemID(sc.id, sc.reasonOrdinal)
			events = append(events, sc.responsesEm().ReasoningItemAdded(sc.thinkItemID, sc.thinkIndexIn))
			events = append(events, sc.responsesEm().ReasoningPartAdded(sc.thinkItemID, sc.thinkIndexIn))
		}
		sc.thinkBuf.WriteString(thinking)
		events = append(events, sc.responsesEm().ReasoningSummaryDelta(sc.thinkItemID, sc.thinkIndexIn, thinking))
	}
	if choice.Delta.Content != "" {
		changed = true
		events = append(events, sc.stopReasoning()...)
		if sc.msgIndex < 0 || sc.textClosed {
			sc.msgIndex = sc.nextIndex
			sc.nextIndex++
			sc.msgItemID = "msg_" + sc.id + "_" + strconv.Itoa(sc.msgIndex)
			sc.textClosed = false
			sc.textPartOpen = false
			sc.respText.Reset()
			events = append(events, sc.responsesEm().ItemAdded(sc.msgIndex, map[string]any{
				"type": "message", "role": "assistant", "id": sc.msgItemID, "status": "in_progress", "content": []any{},
			}))
		}
		if !sc.textPartOpen {
			sc.textPartOpen = true
			events = append(events, sc.responsesEm().TextPartAdded(sc.msgItemID, sc.msgIndex))
		}
		sc.respText.WriteString(choice.Delta.Content)
		events = append(events, sc.responsesEm().TextDelta(sc.msgItemID, sc.msgIndex, choice.Delta.Content))
	}
	for _, tc := range choice.Delta.ToolCalls {
		if tc.ID != "" || tc.Function.Name != "" || tc.Function.Arguments != "" {
			changed = true
		}
		events = append(events, sc.stopReasoning()...)
		events = append(events, sc.closeResponseText("completed")...)
		t := sc.tools[tc.Index]
		if t != nil && t.responseDone && tc.Function.Arguments != "" {
			return nil, errclass.Translation("tool call received arguments after completion")
		}
		if t == nil {
			events = append(events, sc.stopReasoning()...)
			t = &streamTool{blockIndex: sc.nextIndex, id: tc.ID}
			sc.nextIndex++
			sc.tools[tc.Index] = t
			sc.toolsSeen = true
			sc.toolOrder = append(sc.toolOrder, tc.Index)
		}
		if tc.ID != "" && !t.announced && t.id == "" {
			t.id = tc.ID
		}
		if nameChunk := tc.Function.Name; nameChunk != "" && !t.announced {
			switch {
			case t.name == "":
				t.name = nameChunk
			case nameChunk == t.name:
				// Some providers repeat the complete name on continuation chunks.
			case strings.HasPrefix(nameChunk, t.name):
				// Late complete name after an earlier prefix replaces that prefix.
				t.name = nameChunk
			default:
				t.name += nameChunk
			}
		}
		if tc.Function.Arguments != "" {
			t.args.WriteString(tc.Function.Arguments)
		}
		kind, exact, wait := sc.toolContext.ClassifyStreamedName(t.name)
		if !t.announced && t.name != "" && (exact || !wait) {
			if identity, ok := sc.toolContext.ResolveChatName(t.name); ok {
				t.identity = identity
			}
			t.custom = exact && kind == "custom"
			t.announced = true
			outputName := t.name
			if t.identity.Name != "" {
				outputName = t.identity.Name
			}
			if t.custom {
				item := map[string]any{"type": "custom_tool_call", "call_id": t.id, "name": outputName, "input": "", "status": "in_progress"}
				if t.identity.Namespace != "" {
					item["namespace"] = t.identity.Namespace
				}
				events = append(events, sc.responsesEm().ItemAdded(t.blockIndex, item))
			} else {
				item := map[string]any{"type": "function_call", "call_id": t.id, "name": outputName, "arguments": ""}
				if t.identity.Namespace != "" {
					item["namespace"] = t.identity.Namespace
				}
				events = append(events, sc.responsesEm().ItemAdded(t.blockIndex, item))
			}
		}
		if t.announced && !t.custom && t.argsSent < t.args.Len() {
			pending := t.args.String()[t.argsSent:]
			t.argsSent = t.args.Len()
			events = append(events, sc.responsesEm().ArgsDelta(t.id, t.blockIndex, pending))
		}
	}
	if choice.FinishReason != "" && !sc.finished {
		changed = true
		for _, idx := range sc.toolOrder {
			t := sc.tools[idx]
			if !t.announced {
				// An unresolved prefix at finish cannot be proven custom. Preserve it
				// conservatively as a function and flush every buffered argument byte.
				t.announced = true
				events = append(events, sc.responsesEm().ItemAdded(t.blockIndex, map[string]any{
					"type": "function_call", "call_id": t.id, "name": t.name, "arguments": "",
				}))
			}
			if t.custom {
				if _, eErr := shared.UnwrapCustomToolInput(t.args.String()); eErr != nil {
					return nil, eErr
				}
			} else if t.argsSent < t.args.Len() {
				pending := t.args.String()[t.argsSent:]
				t.argsSent = t.args.Len()
				events = append(events, sc.responsesEm().ArgsDelta(t.id, t.blockIndex, pending))
			}
			if !t.custom {
				var decoded any
				if json.Unmarshal([]byte(shared.DefaultArgs(t.args.String())), &decoded) != nil {
					return nil, errclass.Translation("tool call arguments were incomplete at stream completion")
				}
			}
		}
		sc.finished = true
		sc.heldFinish = choice.FinishReason
		sc.streamState.Finish()
		events = append(events, sc.stopReasoning()...)
	}
	if changed && !sc.streamState.Snapshot.Started && sc.started {
		sc.streamState.Advance()
	} else if changed && choice.FinishReason == "" && sc.started {
		sc.streamState.Advance()
	}
	return events, nil
}

// stopReasoning emits the reasoning item's closing transitions exactly
// once, after the summary text is complete: text done, part done, item
// done. Text, tool calls, finish_reason, and [DONE] all trigger it, so a
// thinking run always closes before the next item is announced
// (host-translator parity).
func (sc *StreamConverter) stopReasoning() [][]byte {
	if !sc.thinkSeen || sc.thinkStopped {
		return nil
	}
	sc.thinkStopped = true
	text := sc.thinkBuf.String()
	sc.responseItems[sc.thinkIndexIn] = shared.NewRespReasoningItem(sc.thinkItemID, text)
	em := sc.responsesEm()
	return [][]byte{
		em.ReasoningSummaryDone(sc.thinkItemID, sc.thinkIndexIn, text),
		em.ReasoningPartDone(sc.thinkItemID, sc.thinkIndexIn, text),
		em.ReasoningItemDone(sc.thinkItemID, sc.thinkIndexIn, text),
	}
}

// responsesEm binds the shared Responses emitter kernel to the captured
// upstream chunk identity so this route's frames cannot diverge from the
// sibling Messages-route synthesizer (FR-006).
func (sc *StreamConverter) responsesEm() shared.ResponsesEventEmitter {
	return shared.ResponsesEventEmitter{ID: sc.id, Model: sc.model, Sequence: &sc.respSequence}
}

func (sc *StreamConverter) closeResponseText(status string) [][]byte {
	if sc.msgIndex < 0 || sc.textClosed {
		return nil
	}
	sc.textClosed = true
	content, _ := json.Marshal([]map[string]any{{"type": "output_text", "text": sc.respText.String(), "annotations": []any{}, "logprobs": []any{}}})
	sc.responseItems[sc.msgIndex] = shared.RespItem{Type: "message", ID: sc.msgItemID, Role: "assistant", Status: status, Content: content}
	return sc.responsesEm().TextDone(sc.msgItemID, sc.msgIndex, sc.respText.String(), status)
}

func (sc *StreamConverter) closeResponseTools() ([][]byte, *errclass.Error) {
	var events [][]byte
	for _, idx := range sc.toolOrder {
		t := sc.tools[idx]
		if t == nil || t.responseDone || !t.announced {
			continue
		}
		outputName := t.name
		if t.identity.Name != "" {
			outputName = t.identity.Name
		}
		if t.custom {
			input, eErr := shared.UnwrapCustomToolInput(t.args.String())
			if eErr != nil {
				return nil, eErr
			}
			events = append(events, sc.responsesEm().CustomInputDone(t.id, t.blockIndex, input))
			item := shared.RespItem{Type: "custom_tool_call", CallID: t.id, Name: outputName, Namespace: t.identity.Namespace, Input: input, Status: "completed"}
			events = append(events, sc.responsesEm().ItemDone(t.blockIndex, item))
			sc.responseItems[t.blockIndex] = item
		} else {
			arguments := shared.DefaultArgs(t.args.String())
			var decoded any
			if json.Unmarshal([]byte(arguments), &decoded) != nil {
				continue
			}
			item := shared.RespItem{Type: "function_call", CallID: t.id, Name: outputName, Namespace: t.identity.Namespace, Arguments: arguments, Status: "completed"}
			events = append(events, sc.responsesEm().ArgsDone(t.id, t.blockIndex, item.Arguments, item)...)
			sc.responseItems[t.blockIndex] = item
		}
		t.responseDone = true
	}
	return events, nil
}

// responsesTerminal renders the held response.completed exactly once,
// carrying the captured usage when any arrived before emission.
func (sc *StreamConverter) responsesTerminal(allowSyntheticFinish bool) ([][]byte, *errclass.Error) {
	if sc.terminalSent {
		return nil, nil
	}
	finish := sc.heldFinish
	sc.heldFinish = ""
	if finish == "" {
		if !allowSyntheticFinish {
			return nil, nil
		}
		if !sc.started {
			return nil, errclass.Translation("Responses stream ended without any response events")
		}
		finish = "stop"
	}
	// Status derives positionally from finish alone: length→incomplete,
	// else completed. Tool calls are represented by output items, not
	// status vocabulary (F-R2, Messages-route parity).
	status := shared.ResponseStatusFromCCFinish(finish)
	var terminalEvents [][]byte
	terminalEvents = append(terminalEvents, sc.stopReasoning()...)
	if sc.msgIndex >= 0 && !sc.textClosed {
		itemStatus := "completed"
		if status == "incomplete" {
			itemStatus = "incomplete"
		}
		terminalEvents = append(terminalEvents, sc.closeResponseText(itemStatus)...)
	}
	toolEvents, toolErr := sc.closeResponseTools()
	if toolErr != nil {
		return nil, toolErr
	}
	terminalEvents = append(terminalEvents, toolEvents...)
	for _, idx := range sc.toolOrder {
		t := sc.tools[idx]
		if t != nil && !t.responseDone {
			return nil, errclass.Translation("tool call was incomplete when the stream ended")
		}
	}
	// Always attach (F-R6): zero-valued fields when upstream sent none.
	input, outputTokens := int64(0), int64(0)
	if sc.usage != nil {
		input = sc.usage.PromptTokens
		outputTokens = sc.usage.CompletionTokens
	}
	var details shared.UsageDetails
	if sc.usage != nil {
		if sc.usage.PromptDetails != nil {
			details.CachedTokens = sc.usage.PromptDetails.CachedTokens
		}
		if sc.usage.CompletionDetails != nil {
			details.ReasoningTokens = sc.usage.CompletionDetails.ReasoningTokens
		}
	}
	usage := shared.NewResponsesUsageFrom(input, outputTokens, details)
	output := make([]any, 0, len(sc.responseItems))
	for i := 0; i < sc.nextIndex; i++ {
		if item, ok := sc.responseItems[i]; ok {
			output = append(output, item)
		}
	}
	sc.terminalSent = true
	sc.streamState.Terminal(status)
	return append(terminalEvents, sc.responsesEm().Completed(status, usage, output)), nil
}

// FlushWithError finalizes a Responses stream whose upstream reached EOF
// without [DONE], preserving the same terminal validation as the DONE path.
func (sc *StreamConverter) FlushWithError() ([][]byte, *errclass.Error) {
	if sc.sourceFormat != "openai-response" {
		return sc.FinalizeStream()
	}
	if sc.flushed {
		return nil, nil
	}
	sc.flushed = true
	return sc.responsesTerminal(true)
}

func (sc *StreamConverter) FinalizeStream() ([][]byte, *errclass.Error) {
	if sc.sourceFormat == "openai-response" {
		return sc.responsesTerminal(false)
	}
	if sc.sourceFormat == "openai" || sc.sourceFormat == "claude" {
		if sc.streamState.Snapshot.TerminalSeen {
			return nil, nil
		}
		if !sc.streamState.Snapshot.FinishSeen {
			return nil, errclass.Translation("Chat Completions stream ended before finish_reason")
		}
		if eErr := sc.validateAccumulatedTools(); eErr != nil {
			return nil, eErr
		}
		events := sc.Flush()
		sc.streamState.Terminal("finish")
		return events, nil
	}
	return nil, nil
}
