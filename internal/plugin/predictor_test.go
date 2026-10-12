package plugin

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginabi"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"

	"ollama-cliproxyapi/internal/predictor"
)

func TestPredictorManagementWithoutEngine(t *testing.T) {
	m := NewManager(nil)
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/plugins/" + pluginName + "/prediction",
		Body:   []byte(`{}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	var got predictor.Response
	if err := json.Unmarshal(resp.Body, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Accounts) != 0 {
		t.Fatalf("accounts = %+v", got.Accounts)
	}
	fleet, ok := got.ByType["ollama"]
	if !ok || fleet.TotalAccounts != 0 || fleet.Risk != "" {
		t.Fatalf("fleet = %+v", fleet)
	}
	// D10 visibility: the disabled engine must still say so on the page.
	if got.Persistence == nil || got.Persistence.State != predictor.PersistOff {
		t.Fatalf("persistence = %+v, want off for the disabled engine", got.Persistence)
	}
}

func TestPredictorResourceServed(t *testing.T) {
	m := NewManager(nil)
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{
		Method: http.MethodGet,
		Path:   "/v0/resource/plugins/" + pluginName + "/prediction",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Body) == 0 {
		t.Fatal("prediction page is empty")
	}
}

func TestPredictorManagementUnknownPath(t *testing.T) {
	m := NewManager(nil)
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/plugins/" + pluginName + "/no-such-route",
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

// TestPredictorLifecycle pins the D1 wiring: register builds and starts the
// engine (unless disabled), the store view answers management POSTs with
// zero upstream traffic, and shutdown clears the engine. Upstream silence
// is the hermetic guarantee: the first sample waits a full interval
// (default 10m), far beyond any test lifetime.
func TestPredictorLifecycle(t *testing.T) {
	f := &fakeCaller{}
	m := NewManager(NewHostBridge(f.call))

	// Explicitly disabled → no engine, snapshot stays empty-shaped.
	if _, err := m.HandleCall(pluginabi.MethodPluginRegister, lifecycleRequestBody(testValidYAML+"predictor:\n  enabled: false\n")); err != nil {
		t.Fatal(err)
	}
	if m.currentPredictor() != nil {
		t.Fatal("engine must be nil when disabled")
	}
	if _, err := m.HandleCall(pluginabi.MethodPluginShutdown, []byte(`{}`)); err != nil {
		t.Fatal(err)
	}

	// Default-enabled: engine installs and starts; data-dir off keeps the
	// test hermetic against the developer's real home history file.
	if _, err := m.HandleCall(pluginabi.MethodPluginRegister, lifecycleRequestBody(testValidYAML+"predictor:\n  data-dir: off\n")); err != nil {
		t.Fatal(err)
	}
	pred := m.currentPredictor()
	if pred == nil {
		t.Fatal("engine missing when enabled by default")
	}
	time.Sleep(30 * time.Millisecond)
	// Catalog refresh legitimately rides host.http.do; the predictor's own
	// budget must stay untouched until its first (10m-away) sample.
	if calls := f.callsOf(pluginabi.MethodHostHTTPDo); predictorFetches(calls) > 0 {
		t.Fatalf("register/snapshot triggered predictor upstream calls")
	}
	keyID, _ := quotaIdentity(testKey)

	// Store view: one account card, no windows yet (never sampled).
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/plugins/" + pluginName + "/prediction",
		Body:   []byte(`{}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	var got predictor.Response
	if err := json.Unmarshal(resp.Body, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Accounts) != 1 || got.Accounts[0].KeyID != keyID {
		t.Fatalf("accounts = %+v", got.Accounts)
	}
	fleet, ok := got.ByType["ollama"]
	if !ok || fleet.TotalAccounts != 1 || fleet.ActiveAccounts != 0 {
		t.Fatalf("fleet = %+v", fleet)
	}

	// Manual refresh is accepted and queued (in-budget; sampled by the loop
	// later — zero synchronous upstream traffic).
	if _, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/plugins/" + pluginName + "/prediction",
		Body:   []byte(`{"refresh":true,"key_id":"` + keyID + `"}`),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/plugins/" + pluginName + "/prediction",
		Body:   []byte(`{"refresh":true,"key_id":"ghost"}`),
	}); err != nil {
		t.Fatal(err)
	}

	// Invalid deadlines are rejected before touching the engine.
	resp, err = m.HandleManagement(context.Background(), pluginapi.ManagementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/plugins/" + pluginName + "/prediction",
		Body:   []byte(`{"deadline":"not-a-time"}`),
	})
	if err != nil || resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad deadline: %d %v", resp.StatusCode, err)
	}
	past := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	resp, err = m.HandleManagement(context.Background(), pluginapi.ManagementRequest{
		Method: http.MethodPost,
		Path:   "/v0/management/plugins/" + pluginName + "/prediction",
		Body:   []byte(`{"deadline":"` + past + `"}`),
	})
	if err != nil || resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("past deadline: %d %v", resp.StatusCode, err)
	}

	// Shutdown stops and clears the engine.
	if _, err := m.HandleCall(pluginabi.MethodPluginShutdown, []byte(`{}`)); err != nil {
		t.Fatal(err)
	}
	if m.currentPredictor() != nil {
		t.Fatal("engine must be cleared at shutdown")
	}
	if calls := f.callsOf(pluginabi.MethodHostHTTPDo); predictorFetches(calls) > 0 {
		t.Fatalf("lifecycle produced predictor upstream calls")
	}
}

// predictorFetches counts bridge calls aimed at the Ollama account API
// (the predictor's only upstream surface), ignoring catalog refreshes.
func predictorFetches(calls []capturedCall) int {
	n := 0
	for _, c := range calls {
		var wire map[string]any
		if err := json.Unmarshal(c.payload, &wire); err != nil {
			continue
		}
		if url, ok := wire["url"].(string); ok && strings.Contains(url, "/api/balance") {
			n++
		}
		if url, ok := wire["url"].(string); ok && strings.Contains(url, "/api/usage") {
			n++
		}
	}
	return n
}
