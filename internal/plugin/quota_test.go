package plugin

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginabi"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"

	"ollama-cliproxyapi/internal/config"
	"ollama-cliproxyapi/resources"
)

func TestManagementRegistration(t *testing.T) {
	m := NewManager(nil)
	var got struct {
		Routes    []struct{ Method, Path string }            `json:"routes"`
		Resources []struct{ Path, Menu, Description string } `json:"resources"`
	}
	decodeResult(t, mustHandle(t, m, pluginabi.MethodManagementRegister, []byte(`{}`)), &got)
	if len(got.Routes) != 2 {
		t.Fatalf("routes = %+v", got.Routes)
	}
	if got.Routes[0].Method != http.MethodPost || got.Routes[0].Path != "/plugins/"+pluginName+"/quota-usage" {
		t.Fatalf("route[0] = %+v", got.Routes[0])
	}
	if got.Routes[1].Method != http.MethodPost || got.Routes[1].Path != "/plugins/"+pluginName+"/prediction" {
		t.Fatalf("route[1] = %+v", got.Routes[1])
	}
	if len(got.Resources) != 2 {
		t.Fatalf("resources = %+v", got.Resources)
	}
	if got.Resources[0].Path != "/quota" || got.Resources[0].Menu != "Ollama Quota" {
		t.Fatalf("resource[0] = %+v", got.Resources[0])
	}
	if got.Resources[1].Path != "/prediction" || got.Resources[1].Menu != "Ollama Usage Prediction" {
		t.Fatalf("resource[1] = %+v", got.Resources[1])
	}
	var registration registrationResult
	decodeResult(t, mustHandle(t, m, pluginabi.MethodPluginRegister, lifecycleRequestBody(testValidYAML)), &registration)
	if !registration.Capabilities.ManagementAPI {
		t.Fatal("registration did not advertise management_api")
	}
}

func TestQuotaListDoesNotCallHost(t *testing.T) {
	f := &fakeCaller{}
	m := NewManager(NewHostBridge(f.call))
	m.cfg = config.Config{APIKeys: []config.APIKey{{Value: "quota-key-a"}, {Value: "quota-key-b"}}}
	var got quotaList
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{Method: http.MethodPost, Path: "/v0/management/plugins/" + pluginName + "/quota-usage", Body: []byte(`{}`)})
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(resp.Body, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Cards) != 2 || got.Cards[0].Label != "Ollama credential "+mustHashPrefix("quota-key-a") || got.Cards[1].Label != "Ollama credential "+mustHashPrefix("quota-key-b") {
		t.Fatalf("cards = %+v", got.Cards)
	}
	if strings.Contains(string(resp.Body), "quota-key-") || len(f.callsOf(pluginabi.MethodHostHTTPDo)) != 0 {
		t.Fatalf("list leaked key or called upstream: %s calls=%v", resp.Body, f.callsOf(pluginabi.MethodHostHTTPDo))
	}
}

func TestQuotaRefresh(t *testing.T) {
	const key = "quota-refresh-secret"
	seen := map[string]int{}
	f := &fakeCaller{responder: func(method string, payload []byte) ([]byte, error) {
		if method != pluginabi.MethodHostHTTPDo {
			return hostOK(map[string]any{}), nil
		}
		var wire struct {
			Method  string      `json:"method"`
			URL     string      `json:"url"`
			Headers http.Header `json:"headers"`
		}
		if err := json.Unmarshal(payload, &wire); err != nil || wire.Method != http.MethodGet || wire.Headers.Get("Authorization") != "Bearer "+key || wire.Headers.Get("Accept") != "application/json" {
			t.Fatalf("bad quota request: %s", payload)
		}
		seen[wire.URL]++
		switch {
		case wire.URL == "https://quota.test/api/balance":
			return hostOK(pluginapi.HTTPResponse{StatusCode: http.StatusOK, Body: []byte(
				`{"included":{"balance_usd":72.5,"allowance_usd":100,"period":{"from":"2026-10-01T00:00:00Z","until":"2026-10-31T00:00:00Z"}},"purchased":{"balance_usd":25}}`)}), nil
		case strings.HasPrefix(wire.URL, "https://quota.test/api/usage?"):
			if !strings.Contains(wire.URL, "range=24h") || !strings.Contains(wire.URL, "scope=self") {
				t.Fatalf("usage query = %q", wire.URL)
			}
			return hostOK(pluginapi.HTTPResponse{StatusCode: http.StatusOK, Body: []byte(
				`{"range":"24h","scope":"self","granularity":"hour","totals":{"request_count":12,"usage_usd":3.4,"input_tokens":1500,"cached_input_tokens":500,"output_tokens":2300}}`)}), nil
		}
		t.Fatalf("unexpected quota URL %q", wire.URL)
		return nil, nil
	}}
	m := NewManager(NewHostBridge(f.call))
	// base-url is the bare Ollama authority; the account calls must land on
	// the same authority at /api/*.
	m.cfg = config.Config{BaseURL: "https://quota.test", RequestTimeout: config.DefaultRequestTimeout, APIKeys: []config.APIKey{{Value: key}}}
	id, _ := quotaIdentity(key)
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{Method: http.MethodPost, Path: "/v0/management/plugins/" + pluginName + "/quota-usage", Body: []byte(`{"key_id":"` + id + `"}`)})
	if err != nil {
		t.Fatal(err)
	}
	var got quotaCard
	if err := json.Unmarshal(resp.Body, &got); err != nil {
		t.Fatal(err)
	}
	if got.Error != "" {
		t.Fatalf("unexpected card error: %q", got.Error)
	}
	if got.Label != "Ollama credential "+mustHashPrefix(key) {
		t.Errorf("label = %q", got.Label)
	}
	if got.Usage == nil {
		t.Fatal("usage missing")
	}
	if len(got.Usage.Rows) != 1 {
		t.Fatalf("rows = %+v", got.Usage.Rows)
	}
	row := got.Usage.Rows[0]
	if row.Name != "included" || row.Cap != 100 || row.Used != 27.5 || row.Percent != 27 || row.Status != "ok" {
		t.Errorf("included row = %+v", row)
	}
	if want := "2026-10-31T00:00:00Z"; row.ResetsAt != want {
		t.Errorf("included reset = %q, want %q", row.ResetsAt, want)
	}
	if row.Text != "$72.50 left of $100.00" {
		t.Errorf("row text = %q", row.Text)
	}
	// Fact lines: balance + purchased balance + usage totals.
	if len(got.Usage.Metrics) != 3 {
		t.Fatalf("metrics = %+v", got.Usage.Metrics)
	}
	if got.Usage.Metrics[0].Label != "Balance" || got.Usage.Metrics[0].Value != "$72.50 included" {
		t.Errorf("balance metric = %+v", got.Usage.Metrics[0])
	}
	if got.Usage.Metrics[1].Label != "Purchased balance" || got.Usage.Metrics[1].Value != "$25.00" {
		t.Errorf("purchased metric = %+v", got.Usage.Metrics[1])
	}
	usageLine := got.Usage.Metrics[2]
	if usageLine.Label != "Usage (24h)" || !strings.Contains(usageLine.Value, "12 requests") || !strings.Contains(usageLine.Value, "$3.40") {
		t.Errorf("usage metric = %+v", usageLine)
	}
	if !strings.Contains(usageLine.Value, "1.5k in") || !strings.Contains(usageLine.Value, "500 cached") || !strings.Contains(usageLine.Value, "2.3k out") {
		t.Errorf("usage token fragment = %q", usageLine.Value)
	}
	if got.Usage.RefreshedAt == "" {
		t.Error("refreshed_at missing")
	}
	for url, calls := range seen {
		if calls != 1 {
			t.Errorf("%s called %d times, want 1", url, calls)
		}
	}
	if len(seen) != 2 {
		t.Errorf("endpoints called = %v", seen)
	}
}

func TestQuotaRefreshLegacyPercentShape(t *testing.T) {
	const key = "quota-legacy-secret"
	f := &fakeCaller{responder: func(method string, payload []byte) ([]byte, error) {
		if method != pluginabi.MethodHostHTTPDo {
			return hostOK(map[string]any{}), nil
		}
		var wire struct {
			URL string `json:"url"`
		}
		_ = json.Unmarshal(payload, &wire)
		if wire.URL == "https://quota.test/api/balance" {
			// Legacy plan shape: rolling windows as remaining percentages.
			return hostOK(pluginapi.HTTPResponse{StatusCode: http.StatusOK, Body: []byte(
				`{"included":{"session":{"remaining_percent":40,"resets_at":"2026-10-09T20:00:00Z"},"weekly":{"remaining_percent":0,"resets_at":"2026-10-13T00:00:00Z"}}}`)}), nil
		}
		if strings.HasPrefix(wire.URL, "https://quota.test/api/usage?") {
			// Usage is best-effort: an upstream failure must not blank the card.
			return hostOK(pluginapi.HTTPResponse{StatusCode: http.StatusTooManyRequests, Body: []byte(`{"error":"rate limited"}`)}), nil
		}
		t.Fatalf("unexpected quota URL %q", wire.URL)
		return nil, nil
	}}
	m := NewManager(NewHostBridge(f.call))
	m.cfg = config.Config{BaseURL: "https://quota.test", RequestTimeout: config.DefaultRequestTimeout, APIKeys: []config.APIKey{{Value: key}}}
	id, _ := quotaIdentity(key)
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{Method: http.MethodPost, Path: "/v0/management/plugins/" + pluginName + "/quota-usage", Body: []byte(`{"key_id":"` + id + `"}`)})
	if err != nil {
		t.Fatal(err)
	}
	var got quotaCard
	if err := json.Unmarshal(resp.Body, &got); err != nil {
		t.Fatal(err)
	}
	if got.Error != "" || got.Usage == nil {
		t.Fatalf("card = %+v", got)
	}
	if len(got.Usage.Rows) != 2 {
		t.Fatalf("rows = %+v", got.Usage.Rows)
	}
	session, weekly := got.Usage.Rows[0], got.Usage.Rows[1]
	if session.Name != "session" || session.Percent != 60 || session.Status != "ok" || session.Text != "40% remaining" {
		t.Errorf("session row = %+v", session)
	}
	if session.ResetsAt != "2026-10-09T20:00:00Z" {
		t.Errorf("session reset = %q", session.ResetsAt)
	}
	if weekly.Name != "weekly" || weekly.Status != "exceeded" || weekly.Percent != 100 {
		t.Errorf("weekly row = %+v", weekly)
	}
	// The failed usage call must not produce a usage metric.
	for _, metric := range got.Usage.Metrics {
		if strings.HasPrefix(metric.Label, "Usage") {
			t.Errorf("usage metric leaked from a 429: %+v", metric)
		}
	}
}

func TestQuotaRefreshLocalBaseIsRejected(t *testing.T) {
	const key = "quota-local-secret"
	f := &fakeCaller{}
	m := NewManager(NewHostBridge(f.call))
	m.cfg = config.Config{BaseURL: "http://localhost:11434", AllowHTTP: true, RequestTimeout: config.DefaultRequestTimeout, APIKeys: []config.APIKey{{Value: key}}}
	id, _ := quotaIdentity(key)
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{Method: http.MethodPost, Path: "/v0/management/plugins/" + pluginName + "/quota-usage", Body: []byte(`{"key_id":"` + id + `"}`)})
	if err != nil {
		t.Fatal(err)
	}
	var got quotaCard
	if err := json.Unmarshal(resp.Body, &got); err != nil {
		t.Fatal(err)
	}
	if got.Usage != nil || got.Error == "" || !strings.Contains(got.Error, "local") {
		t.Fatalf("card = %+v, want a local-mode explanation", got)
	}
	if len(f.callsOf(pluginabi.MethodHostHTTPDo)) != 0 {
		t.Fatal("local mode must not call upstream account APIs")
	}
}

func TestPeriodRFC3339(t *testing.T) {
	cases := []struct {
		raw  string
		want string
	}{
		{raw: "2026-10-16T04:03:37.000Z", want: "2026-10-16T04:03:37Z"},
		{raw: "2026-10-16T04:03:37Z", want: "2026-10-16T04:03:37Z"},
		{raw: " 2026-10-16T12:03:37+08:00 ", want: "2026-10-16T04:03:37Z"},
		{raw: "2026-10-16", want: "2026-10-16T00:00:00Z"},
		{raw: "", want: ""},
		{raw: "not-a-date", want: ""},
	}
	for _, tc := range cases {
		if got := periodRFC3339(tc.raw); got != tc.want {
			t.Errorf("periodRFC3339(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}

func TestAccountAPIBase(t *testing.T) {
	cases := map[string]string{
		"https://ollama.com":           "https://ollama.com",
		"https://ollama.com/":          "https://ollama.com",
		"https://ollama.com/v1":        "https://ollama.com",
		"https://mirror.test/v1/":      "https://mirror.test",
		"https://proxy.test/ollama/v1": "https://proxy.test/ollama",
		"http://localhost:11434":       "http://localhost:11434",
		"http://127.0.0.1:11434":       "http://127.0.0.1:11434",
	}
	for in, want := range cases {
		got, err := AccountAPIBase(in)
		if err != nil || got != want {
			t.Errorf("AccountAPIBase(%q) = (%q,%v), want %q", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "not a url", "/api/balance", "ftp://ollama.com"} {
		if got, err := AccountAPIBase(bad); err == nil {
			t.Errorf("AccountAPIBase(%q) = %q, want an error", bad, got)
		}
	}
}

func TestLocalBaseURL(t *testing.T) {
	cases := map[string]bool{
		"http://localhost:11434":     true,
		"http://127.0.0.1:11434":     true,
		"http://box.localhost:11434": true,
		"https://ollama.com":         false,
		"https://api.example.test":   false,
		"http://localhost.evil.com":  false,
	}
	for in, want := range cases {
		if got := localBaseURL(in); got != want {
			t.Errorf("localBaseURL(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestFormatTokens(t *testing.T) {
	cases := map[int64]string{0: "0", 999: "999", 1500: "1.5k", 1234567: "1.2M"}
	for in, want := range cases {
		if got := formatTokens(in); got != want {
			t.Errorf("formatTokens(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestQuotaUnknownKeyAndResource(t *testing.T) {
	const key = "quota-known-secret"
	f := &fakeCaller{}
	m := NewManager(NewHostBridge(f.call))
	m.cfg = config.Config{APIKeys: []config.APIKey{{Value: key}}}
	unknown := "ollama-key-unknown"
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{Method: http.MethodPost, Path: "/v0/management/plugins/" + pluginName + "/quota-usage", Body: []byte(`{"key_id":"` + unknown + `"}`)})
	if err != nil || resp.StatusCode != http.StatusNotFound || strings.Contains(string(resp.Body), key) || strings.Contains(string(resp.Body), unknown) || len(f.callsOf(pluginabi.MethodHostHTTPDo)) != 0 {
		t.Fatalf("unknown response = %+v err=%v calls=%v", resp, err, f.callsOf(pluginabi.MethodHostHTTPDo))
	}
	resource, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{Method: http.MethodGet, Path: "/v0/resource/plugins/" + pluginName + "/quota"})
	if err != nil || resource.StatusCode != 0 || resource.Headers.Get("Content-Type") != "text/html; charset=utf-8" || !strings.Contains(string(resource.Body), "Ollama Quota") || strings.Contains(string(resource.Body), key) {
		t.Fatalf("resource response = %+v err=%v", resource, err)
	}
}

func TestQuotaPageReadsRememberedManagementKey(t *testing.T) {
	if !strings.Contains(resources.QuotaPage, "parsed.state && typeof parsed.state.managementKey === \"string\"") ||
		!strings.Contains(resources.QuotaPage, "return {key: parsed.state.managementKey, status: \"ok\"}") {
		t.Fatal("quota page does not read persisted state.managementKey")
	}
}

func TestQuotaPageDecodesCurrentCPAAuthStorage(t *testing.T) {
	for _, marker := range []string{"enc::v1::", "cli-proxy-api-webui::secure-storage", "TextDecoder"} {
		if !strings.Contains(resources.QuotaPage, marker) {
			t.Fatalf("quota page does not decode current CPA auth storage: missing %q", marker)
		}
	}
}

// Quota failures surface on the card with a redacted reason: one unreachable
// account must not blank the page, and no key material may appear in the
// response.
func TestQuotaErrorsAreRedacted(t *testing.T) {
	const key = "quota-error-secret"
	id, _ := quotaIdentity(key)
	for name, responder := range map[string]func(string, []byte) ([]byte, error){
		"bridge": func(string, []byte) ([]byte, error) { return nil, context.Canceled },
		"status": func(string, []byte) ([]byte, error) {
			return hostOK(pluginapi.HTTPResponse{StatusCode: http.StatusUnauthorized, Body: []byte(key + " upstream body")}), nil
		},
		"json": func(string, []byte) ([]byte, error) {
			return hostOK(pluginapi.HTTPResponse{StatusCode: http.StatusOK, Body: []byte(key + " not json")}), nil
		},
	} {
		t.Run(name, func(t *testing.T) {
			f := &fakeCaller{responder: responder}
			m := NewManager(NewHostBridge(f.call))
			m.cfg = config.Config{BaseURL: "https://quota.test", RequestTimeout: config.DefaultRequestTimeout, APIKeys: []config.APIKey{{Value: key}}}
			resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{Method: http.MethodPost, Path: "/v0/management/plugins/" + pluginName + "/quota-usage", Body: []byte(`{"key_id":"` + id + `"}`)})
			if err != nil || resp.StatusCode != 0 || strings.Contains(string(resp.Body), key) || strings.Contains(string(resp.Body), "upstream body") {
				t.Fatalf("response = %+v err=%v", resp, err)
			}
			var got quotaCard
			if err := json.Unmarshal(resp.Body, &got); err != nil {
				t.Fatal(err)
			}
			if got.Error == "" || got.Usage != nil {
				t.Fatalf("card = %+v, want an error and no usage", got)
			}
			if got.Label != "Ollama credential "+mustHashPrefix(key) {
				t.Errorf("label = %q, want the credential fallback label", got.Label)
			}
		})
	}
}

func TestQuotaUpstreamErrorDiagnosticIsCarried(t *testing.T) {
	const key = "quota-diag-secret"
	id, _ := quotaIdentity(key)
	f := &fakeCaller{responder: func(string, []byte) ([]byte, error) {
		return hostOK(pluginapi.HTTPResponse{StatusCode: http.StatusTooManyRequests, Body: []byte(`{"error":"rate limit exceeded"}`)}), nil
	}}
	m := NewManager(NewHostBridge(f.call))
	m.cfg = config.Config{BaseURL: "https://quota.test", RequestTimeout: config.DefaultRequestTimeout, APIKeys: []config.APIKey{{Value: key}}}
	resp, err := m.HandleManagement(context.Background(), pluginapi.ManagementRequest{Method: http.MethodPost, Path: "/v0/management/plugins/" + pluginName + "/quota-usage", Body: []byte(`{"key_id":"` + id + `"}`)})
	if err != nil {
		t.Fatal(err)
	}
	var got quotaCard
	if err := json.Unmarshal(resp.Body, &got); err != nil {
		t.Fatal(err)
	}
	if got.Error == "" || !strings.Contains(got.Error, "rate limit exceeded") {
		t.Fatalf("card = %+v, want the upstream diagnostic", got)
	}
	if strings.Contains(string(resp.Body), key) {
		t.Fatal("key leaked in card error")
	}
}

func TestQuotaPageIsStaticAndSecretFree(t *testing.T) {
	if resources.QuotaPage == "" || strings.Contains(resources.QuotaPage, testKey) || strings.Contains(resources.QuotaPage, "setInterval") || strings.Contains(resources.QuotaPage, "reset button") {
		t.Fatal("quota page contains a secret, polling, or reset action")
	}
	if !strings.Contains(resources.QuotaPage, "textContent") || !strings.Contains(resources.QuotaPage, "Remember password") {
		t.Fatal("quota page missing safe rendering or login guidance")
	}
}

func TestQuotaPageUsesPersistentQuotaCache(t *testing.T) {
	page := resources.QuotaPage
	for _, marker := range []string{
		`const storageKey = "ollama-cliproxyapi:quota"`,
		"const CACHE_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000",
		"let cacheStorage = \"local\"",
		"let localCache = readCache(localStorage)",
		"if (localCache !== null) cache = localCache",
		"cache = readCache(sessionStorage) || Object.create(null)",
		"localStorage.setItem(storageKey, JSON.stringify(cache))",
		"if (cacheStorage === \"local\") localStorage.setItem(storageKey, JSON.stringify(cache))",
		"sessionStorage.setItem(storageKey, JSON.stringify(cache))",
		"if (cacheStorage === \"local\") { cacheStorage = \"session\"",
		"function readCache(storage)",
		"function validKeyID(keyID)",
		"function cleanUsage(usage)",
		"JSON.parse(stored)",
		"typeof parsed !== \"object\"",
		"Array.isArray(parsed)",
		"Object.create(null)",
		"Object.entries(cache)",
		"values.set(keyID, entry.usage)",
		"cache[keyID] = {usage: safeUsage, fetched_at: timestamp, label: labels.get(keyID) || \"\"}",
		"if (!validKeyID(keyID)) return",
		"delete cache[keyID]",
		"new Set(cards.map(card => card.key_id))",
		// Auto-refresh only uncached cards — the account API is rate limited.
		"cards.filter(card => !values.get(card.key_id))",
		"toLocaleString",
		"Stale · last refreshed ",
		"Quota refresh failed",
	} {
		if !strings.Contains(page, marker) {
			t.Fatalf("quota page missing persistent cache marker %q", marker)
		}
	}
	if strings.Contains(page, "sessionStorage.getItem(storageKey)") {
		t.Fatal("quota page must not use sessionStorage as its primary cache")
	}
	if strings.Index(page, "readCache(localStorage)") > strings.Index(page, "readCache(sessionStorage)") {
		t.Fatal("quota page must read localStorage before sessionStorage")
	}
	if strings.Contains(page, "cache[keyID] = {usage, fetched_at") || strings.Contains(page, "cache[keyID] = {usage: usage") {
		t.Fatal("quota cache marker permits unvalidated usage storage")
	}
	for _, marker := range []string{
		"setTimeout",
		"visibilitychange",
		"document.hidden",
		"window.onfocus",
		"window.addEventListener(\"focus\"",
		"bulk refresh",
	} {
		if strings.Contains(page, marker) {
			t.Fatalf("quota page contains prohibited automatic or expiry behavior %q", marker)
		}
	}
}

func TestQuotaPageUsesNativeQuotaStylesAndThemeBridge(t *testing.T) {
	for _, marker := range []string{
		"quota-page",
		"quota-grid",
		"quota-card",
		"quota-track",
		"quota-fill",
		".quota-fill.is-exceeded { background: var(--error-color); }",
		`fill.className = usage.status === "exceeded" ? "quota-fill is-exceeded" : "quota-fill"`,
		"quota-facts",
		`badge.textContent = "OL"`,
		"repeat(auto-fill, minmax(380px, 1fr))",
		"@media (max-width: 768px)",
		`[data-theme="white"]`,
		`[data-theme="dark"]`,
		`window.parent.document`,
		`data-theme`,
		`--bg-secondary`,
		`frameElement.style.backgroundColor`,
		`frameElement.parentElement.style.backgroundColor`,
		".quota-refresh",
		"--bg-secondary: #faf9f5",
		"--bg-primary: #f0eee8",
		"--bg-tertiary: #e9e6df",
		"--text-primary: #2d2a26",
		"--text-secondary: #6d6760",
		"--text-tertiary: #a29c95",
		"--border-color: #e3e1db",
		"--primary-color: #8b8680",
		"--primary-hover: #7f7a74",
		"border-radius: 8px; padding: 8px 10px",
		`MutationObserver`,
		`catch (_) {}`,
	} {
		if !strings.Contains(resources.QuotaPage, marker) {
			t.Fatalf("quota page missing styling marker %q", marker)
		}
	}
	if strings.Count(resources.QuotaPage, `document.createElement("button")`) != 1 || strings.Contains(resources.QuotaPage, `textContent = "Refresh card"`) || strings.Contains(resources.QuotaPage, "quota-button") || strings.Contains(resources.QuotaPage, "quota-refresh-small") {
		t.Fatal("quota page does not have exactly one per-card refresh button path")
	}
	for _, marker := range []string{"querySelectorAll('head link[rel=\"stylesheet\"], head style')", "cloneNode(true)", "dataset.cpaStyle"} {
		if strings.Contains(resources.QuotaPage, marker) {
			t.Fatalf("quota page still clones parent styles: %q", marker)
		}
	}
}

func mustHashPrefix(key string) string {
	_, label := quotaIdentity(key)
	return strings.TrimPrefix(label, "Ollama credential ")
}
