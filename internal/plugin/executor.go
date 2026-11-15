// Non-stream and stream execution paths (FR-005/FR-006/FR-007, arch §4/§5
// steps 5-8): resolve the public model ID against the catalog snapshot,
// translate the inbound payload to the record's upstream protocol, call
// upstream through the host HTTP callbacks, and translate back to the
// client protocol. Request authentication is selected by CPA.

package plugin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginabi"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"

	"ollama-cliproxyapi/internal/adapter/chatcompletions"
	"ollama-cliproxyapi/internal/adapter/messages"
	"ollama-cliproxyapi/internal/adapter/responses"
	"ollama-cliproxyapi/internal/adapter/shared"
	"ollama-cliproxyapi/internal/catalog"
	"ollama-cliproxyapi/internal/config"
	"ollama-cliproxyapi/internal/errclass"
)

// executorRequest mirrors rpcExecutorRequest: the SDK embeds
// pluginapi.ExecutorRequest untagged, so its fields marshal under Go field
// names ("Model", "SourceFormat", "OriginalRequest", "Stream"). StreamID is
// the DOWNSTREAM host-allocated id for executor.execute_stream emissions;
// ids returned by DoStream are UPSTREAM and never interchangeable (§4).
type executorRequest struct {
	pluginapi.ExecutorRequest
	StreamID string `json:"stream_id,omitempty"`
	// HostCallbackID is the callback context the host opened for THIS request
	// (rpcExecutorRequest.host_callback_id). It is the only handle that ties a
	// host HTTP operation to the client request: the cgo host-callback context
	// itself carries no request scope, so an operation opened without this id
	// is parented to context.Background() and a client abort never reaches it.
	HostCallbackID string `json:"host_callback_id,omitempty"`
}

// resolvedExecution carries everything both execution paths need after
// model/key resolution succeeded.
type resolvedExecution struct {
	cfg      config.Config
	rec      catalog.ModelRecord
	route    catalog.Route
	endpoint string
	key      string
}

// resolveExecution resolves the requested model against the snapshot and
// extracts the key selected by CPA. A non-nil second return is a ready-made
// failure envelope.
func (m *Manager) resolveExecution(req executorRequest) (*resolvedExecution, []byte) {
	m.mu.RLock()
	cfg, mgr := m.cfg, m.mgr
	m.mu.RUnlock()
	if req.AuthProvider != ProviderID {
		return nil, classEnvelope(&errclass.Error{Class: errclass.ClassAuth, Message: "selected auth provider is not ollama"})
	}
	key := strings.TrimSpace(req.AuthAttributes["api_key"])
	debugTrace("executor auth model_len=%d auth_id_present=%t provider_present=%t attr_api_key_present=%t attr_count=%d storage_json_bytes=%d", len(req.Model), req.AuthID != "", req.AuthProvider != "", key != "", len(req.AuthAttributes), len(req.StorageJSON))
	if key == "" {
		return nil, classEnvelope(&errclass.Error{Class: errclass.ClassAuth, Message: "selected auth has no api key"})
	}
	var rec catalog.ModelRecord
	var found bool
	if mgr != nil && req.Model != "" {
		rec, found = mgr.Lookup(req.Model)
	}
	if !found {
		return nil, classEnvelope(&errclass.Error{
			Class:      errclass.ClassInvalidModel,
			Message:    "model not in routable catalog",
			StatusCode: http.StatusNotFound,
		})
	}
	route, endpoint := catalog.ResolveRoute(rec, req.SourceFormat)
	return &resolvedExecution{cfg: cfg, rec: rec, route: route, endpoint: endpoint, key: key}, nil
}

// handleExecute implements executor.execute (non-stream). Stream-flagged
// requests are routed to the stream path instead of rejected.
func (m *Manager) handleExecute(request []byte) ([]byte, error) {
	nonStreamStart := time.Now()
	var req executorRequest
	if err := json.Unmarshal(request, &req); err != nil {
		return ErrEnvelope("invalid_request", "malformed executor request body"), nil
	}
	debugTrace("executor invoked stream=%t stream_id_present=%t model_len=%d source_format_present=%t original_request_bytes=%d payload_bytes=%d", req.Stream, req.StreamID != "", len(req.Model), req.SourceFormat != "", len(req.OriginalRequest), len(req.Payload))
	if req.Stream {
		return m.executeStream(req)
	}
	res, failEnv := m.resolveExecution(req)
	if res == nil {
		return failEnv, nil
	}
	upstreamBody, eErr := buildUpstreamRequest(res.route, res.rec.UpstreamID, req.SourceFormat, req.OriginalRequest, res.rec.Thinking, res.cfg.ResponsesCompatibility)
	if eErr != nil {
		return classEnvelope(eErr), nil
	}

	url := catalog.JoinUpstreamURL(res.cfg.BaseURL, res.endpoint)
	debugTrace("executor resolved public_model_len=%d upstream_model_len=%d route=%s url=%s key_count=%d", len(req.Model), len(res.rec.UpstreamID), res.route, url, len(res.cfg.APIKeys))
	debugTrace("executor sending non-stream url=%s body_len=%d", url, len(upstreamBody))
	ctx, cancel := context.WithTimeout(context.Background(), res.cfg.RequestTimeout)
	defer cancel()
	resp, err := m.bridge.Do(ctx, pluginapi.HTTPRequest{
		Method:  http.MethodPost,
		URL:     url,
		Headers: upstreamAuthHeaders(res.route, res.key, false),
		Body:    upstreamBody,
	})
	if err != nil {
		debugTrace("executor non-stream network error: %v", err)
		return classEnvelope(errclass.FromNetwork(err)), nil
	}
	debugTrace("executor received non-stream status=%d body_len=%d", resp.StatusCode, len(resp.Body))
	if resp.StatusCode >= 400 {
		return classEnvelope(shared.UpstreamStatusErrorWithHeaders(resp.StatusCode, resp.Body, resp.Headers)), nil
	}
	// Parse/envelope guard only; true OOM prevention belongs to the host transport's byte cap.
	if int64(len(resp.Body)) > res.cfg.MaxResponseBytes {
		return classEnvelope(errclass.Translation("response exceeds max-response-bytes")), nil
	}
	converted, eErr := convertNonStream(res.route, req.SourceFormat, resp.StatusCode, resp.Body, responsesRequestContext(req))
	if eErr != nil {
		m.logNonStreamFinished(res.route, req.SourceFormat, "conversion", nonStreamStart, resp.StatusCode, len(resp.Body), nil)
		return classEnvelope(eErr), nil
	}
	present := responseUsagePresent(converted)
	m.logNonStreamFinished(res.route, req.SourceFormat, "completed", nonStreamStart, resp.StatusCode, len(resp.Body), &present)
	debugTrace("executor completed non-stream source_format=%s route=%s response_usage_present=%t", req.SourceFormat, res.route, present)
	return okEnvelope(pluginapi.ExecutorResponse{Payload: converted, Headers: resp.Headers}), nil
}

// logNonStreamFinished emits the non-stream counterpart of the stream finish
// record. usage_present distinguishes "upstream sent no usage member" from
// "sent zeros"; nil means the request did not reach that check (conversion
// failed). The host's log formatter whitelists a fixed field set that does
// not include these diagnostics, so the facts ride in the message body while
// the fields stay for structured consumers.
func (m *Manager) logNonStreamFinished(route catalog.Route, sourceFormat, cause string, start time.Time, status, bodyLen int, usagePresent *bool) {
	if m == nil || m.bridge == nil {
		return
	}
	level := "warn"
	if cause == "completed" {
		level = "info"
	}
	message := fmt.Sprintf("ollama non-stream finished route=%s source_format=%s cause=%s status=%d body_bytes=%d duration_ms=%d", route, sourceFormat, cause, status, bodyLen, time.Since(start).Milliseconds())
	if usagePresent != nil {
		message += fmt.Sprintf(" usage_present=%t", *usagePresent)
	}
	_ = m.bridge.Log(level, message, nil)
}

// responseUsagePresent reports only whether a successful JSON response carries
// a usage member. It is debug-only instrumentation and intentionally does not
// inspect or record response content, prompts, or token values.
func responseUsagePresent(payload []byte) bool {
	var response map[string]json.RawMessage
	if err := json.Unmarshal(payload, &response); err != nil {
		return false
	}
	usage, ok := response["usage"]
	return ok && len(usage) > 0 && string(usage) != "null"
}

func buildUpstreamRequest(route catalog.Route, upstreamModel, sourceFormat string, sourceBody []byte, ts *pluginapi.ThinkingSupport, responsesCompatibility ...string) ([]byte, *errclass.Error) {
	switch route {
	case catalog.RouteChatCompletions:
		return chatcompletions.BuildRequest(upstreamModel, sourceFormat, sourceBody, ts, responsesCompatibility...)
	case catalog.RouteMessages:
		return messages.BuildRequest(upstreamModel, sourceFormat, sourceBody, ts)
	case catalog.RouteResponses:
		return responses.BuildRequest(upstreamModel, sourceFormat, sourceBody, ts)
	}
	return nil, errclass.Translation("unsupported route")
}

// upstreamAuthHeaders builds the FR-007 upstream headers. Stream requests also
// mirror the native OpenAI-compat executor's Accept/Cache-Control pair: OpenAI
// style gateways and CDNs key response buffering off Accept, so a stream
// request without "Accept: text/event-stream" risks server-side buffering
// that presents as slow time-to-first-byte. Content-Type is always set: the
// upstream body is JSON on every route, and a POST body without it is
// malformed per RFC 9110.
func upstreamAuthHeaders(route catalog.Route, key string, stream bool) http.Header {
	var h http.Header
	if route == catalog.RouteMessages {
		// Ollama's Anthropic-compatible endpoint accepts Bearer only.
		h = messages.AuthHeaders(key)
	} else {
		// Chat Completions and Responses endpoints are OpenAI-style bearer.
		h = chatcompletions.AuthHeaders(key)
	}
	h.Set("Content-Type", "application/json")
	if stream {
		h.Set("Accept", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
	}
	return h
}

// convertNonStream routes one upstream response to its adapter's uniform
// translator: every adapter owns status classification (>=400 → §7
// classified errors), native passthrough, and cross-format conversion for
// all client formats.
func convertNonStream(route catalog.Route, sourceFormat string, status int, body []byte, originalRequest ...[]byte) ([]byte, *errclass.Error) {
	switch route {
	case catalog.RouteChatCompletions:
		return chatcompletions.ConvertNonStreamResponse(sourceFormat, status, body, originalRequest...)
	case catalog.RouteMessages:
		return messages.ConvertNonStreamResponse(sourceFormat, status, body)
	case catalog.RouteResponses:
		return responses.ConvertNonStreamResponse(sourceFormat, status, body)
	}
	return nil, errclass.Translation("unsupported route")
}

// classEnvelope renders a classified failure as the wire error envelope;
// ToEnvelopeError redacts and propagates HTTPStatus/RetryAfterSeconds/
// CredentialScoped host-side.
func classEnvelope(e *errclass.Error) []byte {
	wire := errclass.ToEnvelopeError(e)
	out, _ := json.Marshal(pluginabi.Envelope{OK: false, Error: &wire})
	return out
}

// streamConverter is the common shape of the three adapters' stream
// converters: feed one upstream SSE chunk, get translated client events.
type streamConverter interface {
	Feed(chunk []byte) (events [][]byte, done bool, eErr *errclass.Error)
}

// Compile-time proof the close-without-terminal Flush seam (F5) picks up
// both converters that can hold a deferred terminal.
var (
	_ interface{ Flush() [][]byte } = (*chatcompletions.StreamConverter)(nil)
	_ interface{ Flush() [][]byte } = (*messages.StreamConverter)(nil)
)

func newStreamConverter(route catalog.Route, sourceFormat string, originalRequest ...[]byte) streamConverter {
	switch route {
	case catalog.RouteMessages:
		return messages.NewStreamConverter(sourceFormat)
	case catalog.RouteResponses:
		return responses.NewStreamConverter(sourceFormat)
	}
	return chatcompletions.NewStreamConverter(sourceFormat, originalRequest...)
}

// responsesRequestContext returns only an authentic inbound Responses request
// for request-local custom-tool provenance. OriginalRequest wins when it is
// valid JSON; Payload is a compatibility fallback only for Responses callers
// and only when it is itself a valid JSON object. The translated upstream body
// is never considered because its function schema has already erased custom
// tool identity.
func responsesRequestContext(req executorRequest) []byte {
	if req.SourceFormat != "openai-response" {
		return nil
	}
	validObject := func(raw []byte) bool {
		if !json.Valid(raw) {
			return false
		}
		var object map[string]json.RawMessage
		return json.Unmarshal(raw, &object) == nil && object != nil
	}
	if validObject(req.OriginalRequest) {
		return req.OriginalRequest
	}
	if validObject(req.Payload) {
		return req.Payload
	}
	return nil
}

// handleExecuteStream implements executor.execute_stream (FR-006, §7).
// It decodes once and delegates to executeStream so a stream-flagged
// request arriving via executor.execute is never parsed twice.
func (m *Manager) handleExecuteStream(request []byte) ([]byte, error) {
	var req executorRequest
	if err := json.Unmarshal(request, &req); err != nil {
		return ErrEnvelope("invalid_request", "malformed executor request body"), nil
	}
	debugTrace("executor stream invoked stream=%t stream_id_present=%t model_len=%d source_format_present=%t original_request_bytes=%d payload_bytes=%d", req.Stream, req.StreamID != "", len(req.Model), req.SourceFormat != "", len(req.OriginalRequest), len(req.Payload))
	return m.executeStream(req)
}

// executeStream runs the already-decoded stream execution: pre-first-byte errors
// (invalid request, unroutable model, upstream HTTP >=400) return immediately as
// error envelopes so CPA can failover pre-emission.
// On success (upstream HTTP 200 OK), the reading and emitting pump loop runs in
// a background goroutine and executeStream returns okEnvelope immediately so the
// host can start draining chunks to the downstream client without buffer deadlocks.
func (m *Manager) executeStream(req executorRequest) ([]byte, error) {
	streamStart := time.Now()
	res, failEnv := m.resolveExecution(req)
	if res == nil {
		return failEnv, nil
	}
	upstreamBody, eErr := buildUpstreamRequest(res.route, res.rec.UpstreamID, req.SourceFormat, req.OriginalRequest, res.rec.Thinking, res.cfg.ResponsesCompatibility)
	if eErr != nil {
		return classEnvelope(eErr), nil
	}

	url := catalog.JoinUpstreamURL(res.cfg.BaseURL, res.endpoint)
	debugTrace("executor sending stream url=%s body_len=%d", url, len(upstreamBody))
	limits := limitsFromConfig(res.cfg)
	ctx, cancel := context.WithTimeout(context.Background(), limits.first)
	defer cancel()
	// Host HTTP operation: the host runs do_stream under the operation's
	// context, so cancelling it tears the upstream request down the moment the
	// pump exits for any reason — client abort included — instead of letting
	// the pump drain until the idle (3m) / total (30m) watchdog fires.
	//
	// Opened only when this request carries a host callback id, and always on
	// a bridge scoped to it: the host derives the operation's parent context
	// from that callback context (the client request's own context), which is
	// what lets an abort cancel the upstream while no bytes are flowing.
	// Without an id there is no request-scoped parent to inherit, and opening
	// the operation on the LOCAL ctx would be actively harmful — that ctx is
	// cancelled by the defer above the instant this function returns, so it
	// would abort the stream it just started.
	//
	// Never half-applied either: if the host refuses the operation, the bridge
	// stays unscoped, so do_stream goes out exactly as it did before this
	// batch (no operation_id, no host_callback_id) and teardown stays
	// watchdog-bounded.
	opBridge := m.bridge
	operationID := ""
	if m.bridge != nil && req.HostCallbackID != "" {
		scoped := m.bridge.withHostCallbackID(req.HostCallbackID)
		if id, openErr := scoped.OpenOperation(ctx); openErr == nil && id != "" {
			opBridge, operationID = scoped, id
		}
	}
	cancelOp := func() {
		if opBridge != nil {
			_ = opBridge.CancelOperation(operationID)
		}
	}
	st, respHeaders, id, err := opBridge.DoStream(ctx, pluginapi.HTTPRequest{
		Method:  http.MethodPost,
		URL:     url,
		Headers: upstreamAuthHeaders(res.route, res.key, true),
		Body:    upstreamBody,
	}, operationID)
	debugTrace("executor stream DoStream status=%d upstreamID=%s operation_present=%t err=%v", st, id, operationID != "", err)
	if err != nil {
		cancelOp()
		cause := "start_error"
		if errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "timed out") {
			cause = "start_timeout"
		}
		m.logStreamFinished(res.route, req.SourceFormat, cause, streamStart, nil, nil, nil, 0, 0, shared.StreamSnapshot{}, false)
		return classEnvelope(errclass.FromNetwork(err)), nil
	}
	if st >= 400 {
		var body []byte
		if id != "" {
			payload, errMsg, _, readErr := m.bridge.StreamRead(id)
			if readErr == nil {
				if len(payload) > 0 {
					body = payload
				} else if errMsg != "" {
					body = []byte(errMsg)
				}
			}
			_ = m.bridge.StreamClose(id)
		}
		cancelOp()
		m.logStreamFinished(res.route, req.SourceFormat, "http_status", streamStart, nil, nil, nil, int64(len(body)), 0, shared.StreamSnapshot{}, false)
		return classEnvelope(shared.UpstreamStatusErrorWithHeaders(st, body, respHeaders)), nil
	}

	downID := req.StreamID
	if m.bridge != nil {
		m.bridge.inFlight.Add(1)
	}
	go func() {
		if m.bridge != nil {
			defer m.bridge.inFlight.Done()
		}
		// Panic containment: the pump consumes hostile upstream data inside
		// the host process — an unrecovered panic here would abort the whole
		// CPA host, not just this request.
		defer m.recoverStreamPump(downID)
		m.pumpStream(downID, id, cancelOp, res, req.SourceFormat, streamStart, responsesRequestContext(req))
	}()
	return okEnvelope(struct{}{}), nil
}

// recoverStreamPump absorbs a panic escaping the stream pump goroutine and
// surfaces it through host.log. pumpStream's own deferred close (and the
// operation cancel) already released every stream before the panic reached
// this guard, so nothing is re-closed here.
func (m *Manager) recoverStreamPump(downID string) {
	if r := recover(); r == nil {
		return
	}
	if m.bridge == nil {
		return
	}
	_ = m.bridge.Log("error", "stream pump panicked; downstream streams closed",
		map[string]any{"downstream_id": downID})
}

// pumpStream reads the upstream stream, converts each chunk, and emits it
// downstream. cancelOperation, when non-nil, releases the host HTTP operation
// the upstream stream runs under; the deferred call releases it however the
// pump exits.
func (m *Manager) pumpStream(downID, upstreamID string, cancelOperation func(), res *resolvedExecution, sourceFormat string, streamStart time.Time, originalRequest ...[]byte) {
	var closeOnce sync.Once
	// closeStreams terminates both streams. detail, when non-nil, is the
	// structured pluginabi.Error for the downstream close: the host bridge
	// gives it precedence over the plain message, so classified failures
	// keep their HTTP status / retry guidance for the client.
	closeStreams := func(downErrMsg string, detail *pluginabi.Error) {
		closeOnce.Do(func() {
			_ = m.bridge.StreamClose(upstreamID)
			_ = m.bridge.StreamCloseDownstreamDetail(downID, downErrMsg, detail)
		})
	}
	// Deferred teardown on every exit path, including a panic reaching the
	// guard above: close both streams first (closeStreams is Once-guarded, so
	// the body's explicit closes win), then cancel the host operation. Doing
	// the stream close before the operation cancel keeps a normal finish a
	// normal finish — the operation cancel is the parent-context abort that
	// remains for client-abort paths where the pump exits before the upstream
	// would have gone idle.
	defer func() {
		closeStreams("", nil)
		if cancelOperation != nil {
			cancelOperation()
		}
	}()

	conv := newStreamConverter(res.route, sourceFormat, originalRequest...)
	limits := limitsFromConfig(res.cfg)
	controller := newStreamController(m.bridge, upstreamID, streamStart, limits)
	controller.firstData(time.Until(streamStart.Add(limits.first)))
	defer controller.stop()
	type snapshotter interface{ StreamSnapshot() shared.StreamSnapshot }
	snapper, _ := conv.(snapshotter)
	var lastSnapshot shared.StreamSnapshot
	var (
		total          int64
		upstreamClosed bool
		convDone       bool
		chunks         int64
	)
	headersAt := time.Now()
	var firstDataAt, lastProgressAt *time.Time
	type usageTracer interface {
		UsageTrace() chatcompletions.UsageTrace
	}
	tracer, _ := conv.(usageTracer)
	logged := false
	logDone := func(cause string, eof bool) {
		if logged {
			return
		}
		logged = true
		var usageTrace []chatcompletions.UsageTrace
		if tracer != nil {
			usageTrace = []chatcompletions.UsageTrace{tracer.UsageTrace()}
		}
		m.logStreamFinished(res.route, sourceFormat, cause, streamStart, &headersAt, firstDataAt, lastProgressAt, total, chunks, lastSnapshot, eof, usageTrace...)
	}
	finishCause := ""
	for {
		payload, readErrMsg, closed, err := m.bridge.StreamRead(upstreamID)
		debugTrace("executor stream read chunk_len=%d closed=%t readErrMsg=%q err=%v", len(payload), closed, readErrMsg, err)
		upstreamClosed = closed
		if cause := controller.Cause(); cause != "" {
			finishCause = cause
			if cause == "finish_grace" {
				if finalizer, ok := conv.(interface {
					FinalizeStream() ([][]byte, *errclass.Error)
				}); ok {
					events, fErr := finalizer.FinalizeStream()
					if fErr != nil {
						wire := errclass.ToEnvelopeError(fErr)
						closeStreams(wire.Message, &wire)
						logDone("terminal_error", upstreamClosed)
						return
					}
					if emitErr := m.emitAll(downID, events); emitErr != nil {
						closeStreams(errclass.Redact(emitErr.Error()), nil)
						logDone("emit_error", upstreamClosed)
						return
					}
					logDone("finish_grace", upstreamClosed)
					return
				}
				closeStreams("stream terminal incomplete", nil)
				logDone("terminal_error", upstreamClosed)
				return
			}
			closeStreams(streamCauseError(cause), streamWatchdogDetail(cause))
			logDone(cause, upstreamClosed)
			return
		}
		if err != nil {
			closeStreams(errclass.Redact(err.Error()), nil)
			logDone("read_error", upstreamClosed)
			return
		}
		if readErrMsg != "" {
			closeStreams(errclass.Redact(readErrMsg), nil)
			logDone("upstream_error", upstreamClosed)
			return
		}
		total += int64(len(payload))
		if len(payload) > 0 {
			chunks++
		}
		if total > res.cfg.MaxResponseBytes {
			closeStreams(errclass.Redact("stream exceeded max-response-bytes"), nil)
			logDone("max_bytes", upstreamClosed)
			return
		}
		events, done, convErr := conv.Feed(payload)
		if snapper != nil {
			snap := snapper.StreamSnapshot()
			if snap.Progress > lastSnapshot.Progress {
				now := time.Now()
				if firstDataAt == nil {
					firstDataAt = &now
				}
				lastProgressAt = &now
				controller.progress()
				if snap.FinishSeen && !lastSnapshot.FinishSeen {
					controller.finish()
				}
				lastSnapshot = snap
			}
		}
		if convErr != nil {
			wire := errclass.ToEnvelopeError(convErr)
			closeStreams(wire.Message, &wire)
			logDone("terminal_error", upstreamClosed)
			return
		}
		if emitErr := m.emitAll(downID, events); emitErr != nil {
			closeStreams(errclass.Redact(emitErr.Error()), nil)
			logDone("emit_error", upstreamClosed)
			return
		}
		convDone = done
		if convDone || upstreamClosed {
			break
		}
	}
	debugTrace("executor stream loop end convDone=%t upstreamClosed=%t", convDone, upstreamClosed)
	if !convDone && upstreamClosed {
		if flusher, ok := conv.(interface {
			FlushWithError() ([][]byte, *errclass.Error)
		}); ok {
			flushed, flushErr := flusher.FlushWithError()
			if flushErr != nil {
				wire := errclass.ToEnvelopeError(flushErr)
				closeStreams(wire.Message, &wire)
				logDone("truncated", true)
				return
			}
			if emitErr := m.emitAll(downID, flushed); emitErr != nil {
				closeStreams(errclass.Redact(emitErr.Error()), nil)
				logDone("emit_error", true)
				return
			}
		} else if flusher, ok := conv.(interface{ Flush() [][]byte }); ok {
			flushed := flusher.Flush()
			if emitErr := m.emitAll(downID, flushed); emitErr != nil {
				closeStreams(errclass.Redact(emitErr.Error()), nil)
				logDone("emit_error", true)
				return
			}
		}
	}
	if !logged && finishCause == "" {
		finishCause = "terminal"
		if upstreamClosed {
			finishCause = "eof"
		}
		logDone(finishCause, upstreamClosed)
	}
}

func streamCauseError(cause string) string {
	switch cause {
	case "first_data":
		return "stream first data timeout"
	case "idle":
		return "stream idle timeout"
	case "total":
		return "stream total timeout"
	case "finish_grace":
		return "stream terminal incomplete"
	}
	return "stream truncated"
}

// streamWatchdogDetail renders a watchdog cause as the structured
// error_detail for the downstream close. Code is pluginabi's
// ErrorCodeStreamTruncated: the upstream generation was already accepted and
// partially delivered, so v8 must skip credential cooldown and must not replay
// the request on another credential, while the client still sees the timeout
// as a 504 rather than the host's status-less default. Unknown causes return
// nil, leaving the plain message only.
func streamWatchdogDetail(cause string) *pluginabi.Error {
	switch cause {
	case "first_data", "idle", "total":
		return &pluginabi.Error{
			Code:       pluginabi.ErrorCodeStreamTruncated,
			Message:    streamCauseError(cause),
			HTTPStatus: http.StatusGatewayTimeout,
		}
	}
	return nil
}

func (m *Manager) logStreamFinished(route catalog.Route, sourceFormat, cause string, start time.Time, headersAt, firstDataAt, lastProgressAt *time.Time, total, chunks int64, snapshot shared.StreamSnapshot, eof bool, streamUsage ...StreamUsageTrace) {
	if m == nil || m.bridge == nil {
		return
	}
	fields := map[string]any{"route": string(route), "source_format": sourceFormat, "cause": cause, "duration_ms": time.Since(start).Milliseconds(), "bytes": total, "chunks": chunks, "finish_seen": snapshot.FinishSeen, "done_seen": snapshot.DoneSeen, "eof": eof, "headers_ms": nil, "first_data_ms": nil, "last_progress_ms": nil}
	if headersAt != nil {
		fields["headers_ms"] = headersAt.Sub(start).Milliseconds()
	}
	if firstDataAt != nil {
		fields["first_data_ms"] = firstDataAt.Sub(start).Milliseconds()
	}
	if lastProgressAt != nil {
		fields["last_progress_ms"] = lastProgressAt.Sub(start).Milliseconds()
	}
	if len(streamUsage) > 0 {
		fields["usage_trailer_seen"] = streamUsage[0].Seen
		fields["usage_input_tokens"] = streamUsage[0].InputTokens
		fields["usage_output_tokens"] = streamUsage[0].OutputTokens
	}
	level := "warn"
	if cause == "terminal" || cause == "eof" || cause == "finish_grace" {
		level = "info"
	}
	// The host's log formatter whitelists a fixed field set that does not
	// include these diagnostics, so the key facts ride in the message body
	// (never filtered) while the fields stay for structured consumers.
	message := fmt.Sprintf("ollama stream finished route=%s source_format=%s cause=%s duration_ms=%d bytes=%d chunks=%d finish_seen=%t done_seen=%t eof=%t", route, sourceFormat, cause, fields["duration_ms"], total, chunks, snapshot.FinishSeen, snapshot.DoneSeen, eof)
	if len(streamUsage) > 0 {
		message += fmt.Sprintf(" usage_trailer_seen=%t usage_input_tokens=%d usage_output_tokens=%d", streamUsage[0].Seen, streamUsage[0].InputTokens, streamUsage[0].OutputTokens)
	}
	_ = m.bridge.Log(level, message, fields)
}

// StreamUsageTrace mirrors the usage captured by the Chat Completions
// stream converter so the finish log can distinguish "upstream sent no
// usage" (trailer never arrived) from "upstream sent zeros" — the two
// causes a manager showing 0/0 tokens needs telling apart. Debug-only
// instrumentation; token values are metadata, not response content.
type StreamUsageTrace = chatcompletions.UsageTrace

// emitAll feeds converted events downstream in order. StreamEmit is
// deadline-bounded (emitTimeout), so a host that stops draining the
// downstream stream surfaces here as an error; the pump loop treats that
// as a post-first-byte stream-fatal failure and runs fail(), whose
// Once-closer releases both streams.
func (m *Manager) emitAll(downStreamID string, events [][]byte) error {
	for _, evt := range events {
		if err := m.bridge.StreamEmit(downStreamID, evt); err != nil {
			return err
		}
	}
	return nil
}
