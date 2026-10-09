package plugin

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginabi"
)

type timedRead struct {
	after   time.Duration
	payload string
	done    bool
}

// timedStreamHost is a channel-driven responder: StreamClose releases a read
// immediately, so timeout tests never leave a fake host callback parked.
type timedStreamHost struct {
	mu       sync.Mutex
	reads    []timedRead
	next     int
	closed   chan struct{}
	closeOne sync.Once
}

func newTimedStreamHost(reads ...timedRead) *timedStreamHost {
	return &timedStreamHost{reads: reads, closed: make(chan struct{})}
}

func (h *timedStreamHost) respond(method string, _ []byte) ([]byte, error) {
	switch method {
	case pluginabi.MethodHostHTTPDoStream:
		return hostOK(hostStreamStartResp{StatusCode: http.StatusOK, StreamID: "timed-up"}), nil
	case pluginabi.MethodHostHTTPStreamRead:
		h.mu.Lock()
		if h.next >= len(h.reads) {
			h.mu.Unlock()
			<-h.closed
			return hostOK(hostStreamReadResp{Done: true}), nil
		}
		r := h.reads[h.next]
		h.next++
		h.mu.Unlock()
		timer := time.NewTimer(r.after)
		defer timer.Stop()
		select {
		case <-timer.C:
			return hostOK(hostStreamReadResp{Payload: []byte(r.payload), Done: r.done}), nil
		case <-h.closed:
			return hostOK(hostStreamReadResp{Done: true}), nil
		}
	case pluginabi.MethodHostHTTPStreamClose:
		h.closeOne.Do(func() { close(h.closed) })
		return hostOK(map[string]any{}), nil
	default:
		return hostOK(map[string]any{}), nil
	}
}

func startTimedStream(t *testing.T, h *timedStreamHost, settings, request, key string) (*Manager, *fakeCaller) {
	t.Helper()
	f := &fakeCaller{}
	m := NewManager(NewHostBridge(f.call))
	t.Cleanup(func() { _, _ = m.HandleCall("plugin.shutdown", nil) })
	f.responder = wrapWithCatalog(`{"data":[{"id":"glm-5.3"}]}`, h.respond)
	yaml := "api-keys:\n  - value: " + key + "\n" + settings
	if _, err := m.HandleCall("plugin.register", lifecycleRequestBody(yaml)); err != nil {
		t.Fatalf("register: %v", err)
	}
	resp, err := m.HandleCall("executor.execute_stream", execStreamReqBodyForKey("ollama/glm-5.3", "openai", []byte(request), "timed-down", key))
	if err != nil || !decodeEnv(t, resp).OK {
		t.Fatalf("execute_stream start: %s, %v", resp, err)
	}
	if !m.bridge.WaitForInFlight(3 * time.Second) {
		t.Fatal("stream did not terminate")
	}
	return m, f
}

func timeoutSettings(first, idle, total, grace time.Duration) string {
	return "request-timeout: 10ms\n" +
		"stream-first-data-timeout: " + first.String() + "\n" +
		"stream-idle-timeout: " + idle.String() + "\n" +
		"stream-total-timeout: " + total.String() + "\n" +
		"stream-finish-grace: " + grace.String() + "\n"
}

func chatDelta(text string) string {
	b, _ := json.Marshal(map[string]any{"id": "r", "choices": []any{map[string]any{"delta": map[string]any{"content": text}}}})
	return "data: " + string(b) + "\n\n"
}

func finishFrame() string {
	return `data: {"id":"r","choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n"
}

func downstreamClose(t *testing.T, f *fakeCaller) string {
	t.Helper()
	calls := f.callsOf(pluginabi.MethodHostStreamClose)
	if len(calls) != 1 {
		t.Fatalf("downstream closes = %d, want 1", len(calls))
	}
	return string(calls[0].payload)
}

func finishLogs(f *fakeCaller) []capturedCall {
	var out []capturedCall
	for _, c := range f.callsOf(pluginabi.MethodHostLog) {
		if strings.Contains(string(c.payload), "ollama stream finished") {
			out = append(out, c)
		}
	}
	return out
}

func decodeFinishLog(t *testing.T, f *fakeCaller) (string, string) {
	t.Helper()
	logs := finishLogs(f)
	if len(logs) != 1 {
		t.Fatalf("finish logs=%d", len(logs))
	}
	var req struct {
		Level  string         `json:"level"`
		Fields map[string]any `json:"fields"`
	}
	if json.Unmarshal(logs[0].payload, &req) != nil {
		t.Fatal("decode log")
	}
	cause, _ := req.Fields["cause"].(string)
	return req.Level, cause
}

func TestStreamValidProgressOutlivesRequestTimeout(t *testing.T) {
	h := newTimedStreamHost(
		timedRead{after: 35 * time.Millisecond, payload: chatDelta("a")},
		timedRead{after: 35 * time.Millisecond, payload: chatDelta("b")},
		timedRead{after: 35 * time.Millisecond, payload: finishFrame()},
		timedRead{after: 5 * time.Millisecond, payload: "data: [DONE]\n\n"},
	)
	_, f := startTimedStream(t, h, timeoutSettings(100*time.Millisecond, 100*time.Millisecond, 500*time.Millisecond, 30*time.Millisecond), ccRequestBody, testKey)
	if close := downstreamClose(t, f); strings.Contains(close, `"error"`) {
		t.Fatalf("progressing stream failed after old request-timeout: %s", close)
	}
}

func TestStreamFirstDataTimeoutAfterHeaders(t *testing.T) {
	_, f := startTimedStream(t, newTimedStreamHost(), timeoutSettings(80*time.Millisecond, 120*time.Millisecond, 400*time.Millisecond, 20*time.Millisecond), ccRequestBody, testKey)
	if close := downstreamClose(t, f); !strings.Contains(close, "stream first data timeout") {
		t.Fatalf("close = %s", close)
	}
}

func TestStreamCommentsAndPingDoNotRefreshIdle(t *testing.T) {
	h := newTimedStreamHost(
		timedRead{after: 5 * time.Millisecond, payload: chatDelta("start")},
		timedRead{after: 35 * time.Millisecond, payload: ": keepalive\n\n"},
		timedRead{after: 35 * time.Millisecond, payload: "event: ping\ndata: {}\n\n"},
	)
	_, f := startTimedStream(t, h, timeoutSettings(100*time.Millisecond, 80*time.Millisecond, 400*time.Millisecond, 20*time.Millisecond), ccRequestBody, testKey)
	if close := downstreamClose(t, f); !strings.Contains(close, "stream idle timeout") {
		t.Fatalf("keepalive refreshed idle timer: %s", close)
	}
}

func TestStreamProgressThenSilenceHitsIdle(t *testing.T) {
	h := newTimedStreamHost(timedRead{after: 5 * time.Millisecond, payload: chatDelta("progress")})
	_, f := startTimedStream(t, h, timeoutSettings(100*time.Millisecond, 70*time.Millisecond, 400*time.Millisecond, 20*time.Millisecond), ccRequestBody, testKey)
	if close := downstreamClose(t, f); !strings.Contains(close, "stream idle timeout") {
		t.Fatalf("close = %s", close)
	}
}

func TestStreamTotalIsAbsoluteDespiteProgress(t *testing.T) {
	reads := make([]timedRead, 8)
	for i := range reads {
		reads[i] = timedRead{after: 30 * time.Millisecond, payload: chatDelta("x")}
	}
	_, f := startTimedStream(t, newTimedStreamHost(reads...), timeoutSettings(80*time.Millisecond, 80*time.Millisecond, 170*time.Millisecond, 20*time.Millisecond), ccRequestBody, testKey)
	if close := downstreamClose(t, f); !strings.Contains(close, "stream total timeout") {
		t.Fatalf("close = %s", close)
	}
}

func TestStreamFinishGraceCompletesWithoutDoneOrEOF(t *testing.T) {
	h := newTimedStreamHost(
		timedRead{after: 5 * time.Millisecond, payload: chatDelta("ok")},
		timedRead{after: 5 * time.Millisecond, payload: finishFrame()},
	)
	_, f := startTimedStream(t, h, timeoutSettings(100*time.Millisecond, 100*time.Millisecond, 500*time.Millisecond, 40*time.Millisecond), ccRequestBody, testKey)
	if close := downstreamClose(t, f); strings.Contains(close, `"error"`) {
		t.Fatalf("finish grace did not finalize complete response: %s", close)
	}
	if logs := finishLogs(f); len(logs) != 1 || !strings.Contains(string(logs[0].payload), `"cause":"finish_grace"`) {
		t.Fatalf("finish logs = %v", logs)
	}
	if level, cause := decodeFinishLog(t, f); level != "info" || cause != "finish_grace" {
		t.Fatalf("level=%s cause=%s", level, cause)
	}
}

func TestStreamFinishGraceRejectsIncompleteTool(t *testing.T) {
	frame := `data: {"id":"r","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{\"secret\":"}}]},"finish_reason":"tool_calls"}]}` + "\n\n"
	_, f := startTimedStream(t, newTimedStreamHost(timedRead{after: 5 * time.Millisecond, payload: frame}), timeoutSettings(100*time.Millisecond, 100*time.Millisecond, 500*time.Millisecond, 40*time.Millisecond), ccRequestBody, testKey)
	if close := downstreamClose(t, f); !strings.Contains(close, `"error"`) {
		t.Fatalf("incomplete tool falsely succeeded: %s", close)
	}
	for _, emit := range f.callsOf(pluginabi.MethodHostStreamEmit) {
		if strings.Contains(string(emit.payload), "response.completed") {
			t.Fatalf("incomplete tool emitted response.completed: %s", emit.payload)
		}
	}
	if level, cause := decodeFinishLog(t, f); level != "warn" || cause != "terminal_error" {
		t.Fatalf("level=%s cause=%s", level, cause)
	}
}

func TestStreamDoneTimeoutRaceTerminatesExactlyOnce(t *testing.T) {
	for i := 0; i < 12; i++ {
		h := newTimedStreamHost(
			timedRead{after: time.Millisecond, payload: chatDelta("x")},
			timedRead{after: 55 * time.Millisecond, payload: "data: [DONE]\n\n"},
		)
		_, f := startTimedStream(t, h, timeoutSettings(80*time.Millisecond, 55*time.Millisecond, 300*time.Millisecond, 20*time.Millisecond), ccRequestBody, testKey)
		_ = downstreamClose(t, f)
		if got := len(finishLogs(f)); got != 1 {
			t.Fatalf("iteration %d termination logs = %d, want 1", i, got)
		}
	}
}

func TestStreamDiagnosticsExcludeRequestSecrets(t *testing.T) {
	const secret = "fake-secret-prompt-key-header-8491"
	request := `{"model":"x","messages":[{"role":"user","content":"` + secret + `"}]}`
	_, f := startTimedStream(t, newTimedStreamHost(), timeoutSettings(60*time.Millisecond, 100*time.Millisecond, 300*time.Millisecond, 20*time.Millisecond), request, secret)
	if logs := finishLogs(f); len(logs) != 1 {
		t.Fatalf("finish logs = %d", len(logs))
	}
	for _, log := range f.callsOf(pluginabi.MethodHostLog) {
		if strings.Contains(string(log.payload), secret) {
			t.Fatalf("diagnostic leaked request secret: %s", log.payload)
		}
	}
}

func TestStreamEOFAfterFinishHasNoError(t *testing.T) {
	h := newTimedStreamHost(
		timedRead{after: time.Millisecond, payload: chatDelta("ok")},
		timedRead{after: time.Millisecond, payload: finishFrame()},
		timedRead{after: time.Millisecond, done: true},
	)
	_, f := startTimedStream(t, h, timeoutSettings(80*time.Millisecond, 80*time.Millisecond, 300*time.Millisecond, 20*time.Millisecond), ccRequestBody, testKey)
	if close := downstreamClose(t, f); strings.Contains(close, `"error"`) {
		t.Fatalf("EOF after finish returned an error: %s", close)
	}
}
