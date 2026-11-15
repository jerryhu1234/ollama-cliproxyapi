package plugin

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginabi"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"

	"ollama-cliproxyapi/internal/catalog"
	"ollama-cliproxyapi/internal/config"
)

const (
	cooledRefreshKey  = "sk-refresh-cooled-key"
	goodRefreshKey    = "sk-refresh-good-key"
	refreshTwoKeyYAML = "api-keys:\n" +
		"  - value: " + cooledRefreshKey + "\n" +
		"  - value: " + goodRefreshKey + "\n" +
		"catalog-url: https://catalog.test/models\n"
)

// authRecorder answers host.http.do, records every Authorization header, and
// rejects the cooled first key 401-style while any other key gets a good
// catalog body.
type authRecorder struct {
	mu    sync.Mutex
	auths []string
	logs  [][]byte
	// body, when non-empty, replaces testCatalogJSON as the served catalog.
	body []byte
}

func (r *authRecorder) call(method string, payload []byte) ([]byte, error) {
	if method == pluginabi.MethodHostLog {
		r.mu.Lock()
		r.logs = append(r.logs, append([]byte(nil), payload...))
		r.mu.Unlock()
		return hostOK(map[string]any{}), nil
	}
	if method != pluginabi.MethodHostHTTPDo {
		return hostOK(map[string]any{}), nil
	}
	var wire struct {
		Headers http.Header `json:"headers"`
	}
	if err := json.Unmarshal(payload, &wire); err != nil {
		return hostErr("test", "undecodable payload"), nil
	}
	auth := wire.Headers.Get("Authorization")
	r.mu.Lock()
	r.auths = append(r.auths, auth)
	r.mu.Unlock()
	if auth == "Bearer "+cooledRefreshKey {
		return hostOK(pluginapi.HTTPResponse{StatusCode: http.StatusUnauthorized}), nil
	}
	body := testCatalogJSON
	if len(r.body) > 0 {
		body = string(r.body)
	}
	return hostOK(pluginapi.HTTPResponse{
		StatusCode: http.StatusOK,
		Headers:    http.Header{"Content-Type": []string{"application/json"}},
		Body:       []byte(body),
	}), nil
}

func (r *authRecorder) seen() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.auths...)
}

func (r *authRecorder) logCalls() [][]byte {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([][]byte(nil), r.logs...)
}

func twoKeyFixture(t *testing.T) (config.Config, *catalog.Manager, *HostBridge, *authRecorder) {
	t.Helper()
	cfg, err := config.Load([]byte(refreshTwoKeyYAML))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	rec := &authRecorder{}
	bridge := NewHostBridge(rec.call)
	return cfg, catalog.New(cfg, bridge), bridge, rec
}

// TestRefreshFailsOverAcrossConfiguredKeys pins the failover contract: a
// failing first key must NOT pin the catalog — refresh walks every configured
// key and the first success wins. (Deliberately replaces the old
// "first-key-only" pin, which left the whole model list dead whenever that one
// key was revoked.) The per-key failure is logged, and the successful key's
// snapshot is installed.
func TestRefreshFailsOverAcrossConfiguredKeys(t *testing.T) {
	cfg, mgr, bridge, rec := twoKeyFixture(t)
	if err := refreshOnce(context.Background(), mgr, bridge, time.Second, cfg); err != nil {
		t.Fatalf("refreshOnce must succeed via the second key: %v", err)
	}
	auths := rec.seen()
	if len(auths) != 2 || auths[0] != "Bearer "+cooledRefreshKey || auths[1] != "Bearer "+goodRefreshKey {
		t.Fatalf("refresh auths = %v; want first key then second key", auths)
	}
	if len(mgr.Models()) == 0 {
		t.Fatal("failover success installed no models")
	}
	var sawPerKey bool
	for _, raw := range rec.logCalls() {
		if strings.Contains(string(raw), "trying next configured key") {
			sawPerKey = true
		}
	}
	if !sawPerKey {
		t.Fatal("per-key failure must log a warn naming the retry")
	}
}

// TestRefreshWithoutKeysFailsLocally pins the local length guard: config
// validation is a cross-package invariant, so refreshOnce must not index
// APIKeys[0] on a directly-built Config. It errors instead of panicking and
// makes no host call at all.
func TestRefreshWithoutKeysFailsLocally(t *testing.T) {
	cfg, err := config.Load([]byte("api-keys:\n  - value: sk-refresh-good-key\n" +
		"catalog-url: https://catalog.test/models\n"))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	rec := &authRecorder{}
	bridge := NewHostBridge(rec.call)
	mgr := catalog.New(cfg, bridge)
	cfg.APIKeys = nil
	if err := refreshOnce(context.Background(), mgr, bridge, time.Second, cfg); err == nil {
		t.Fatal("refresh without keys must fail")
	}
	if got := len(rec.seen()); got != 0 {
		t.Fatalf("refresh without keys issued %d host calls, want 0", got)
	}
}

// TestRefreshUnsupportedLogQuotesNewlineIDs pins F2: a catalog entry excluded
// from the routable set must reach the host.log "models" field quoted (%q), so
// no raw newline carried by an upstream ID can forge log lines (CWE-117).
// The exclusion is produced by the protocol kill-switch — the default route is
// chat-completions here, so turning that switch off demotes the entry while
// leaving its ID in the diagnostic.
func TestRefreshUnsupportedLogQuotesNewlineIDs(t *testing.T) {
	const evilID = "evil\n2027-01-01 ERROR forged host.log line"
	cfg, err := config.Load([]byte("api-keys:\n  - value: sk-refresh-evil-key\n" +
		"catalog-url: https://catalog.test/models\n" +
		"protocols:\n  chat-completions: false\n"))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	rec := &authRecorder{body: []byte(`{"data":[{"id":"evil\n2027-01-01 ERROR forged host.log line"}]}`)}
	bridge := NewHostBridge(rec.call)
	mgr := catalog.New(cfg, bridge)
	if err := refreshOnce(context.Background(), mgr, bridge, time.Second, cfg); err != nil {
		t.Fatalf("refreshOnce: %v", err)
	}
	logCalls := rec.logCalls()
	if len(logCalls) != 1 {
		t.Fatalf("host.log calls = %d, want exactly one unsupported-models warn", len(logCalls))
	}
	var req struct {
		Level   string         `json:"level"`
		Message string         `json:"message"`
		Fields  map[string]any `json:"fields"`
	}
	if err := json.Unmarshal(logCalls[0], &req); err != nil {
		t.Fatalf("decode host.log payload: %v", err)
	}
	if req.Message != "unsupported models excluded from routable catalog" {
		t.Fatalf("unexpected log message: %q", req.Message)
	}
	modelsField, _ := req.Fields["models"].(string)
	if strings.ContainsRune(modelsField, '\n') {
		t.Fatalf("models log field contains raw newline (log injection): %q", modelsField)
	}
	if want := fmt.Sprintf("%q: ", evilID); !strings.Contains(modelsField, want) {
		t.Fatalf("models field must quote the upstream ID; got %q, want prefix %q", modelsField, want)
	}
}
