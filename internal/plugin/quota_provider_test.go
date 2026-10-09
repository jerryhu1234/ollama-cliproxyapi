package plugin

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginabi"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"

	"ollama-cliproxyapi/internal/config"
)

const quotaProviderSecret = "sk-native-quota-secret"

func quotaProviderFixture(t *testing.T) (*Manager, *fakeCaller) {
	t.Helper()
	f := &fakeCaller{responder: func(method string, payload []byte) ([]byte, error) {
		if method != pluginabi.MethodHostHTTPDo {
			return hostOK(map[string]any{}), nil
		}
		var req hostHTTPReq
		if err := json.Unmarshal(payload, &req); err != nil {
			t.Fatalf("decode host request: %v", err)
		}
		if req.HostCallbackID != "quota-callback" {
			t.Fatalf("host_callback_id = %q", req.HostCallbackID)
		}
		if got := req.Headers.Get("Authorization"); got != "Bearer "+quotaProviderSecret {
			t.Fatalf("authorization header was not derived from credential context")
		}
		var body string
		switch {
		case strings.HasSuffix(req.URL, "/api/balance"):
			body = `{"included":{"balance_usd":72.5,"allowance_usd":100,"period":{"from":"2026-10-01T00:00:00Z","until":"2026-10-31T00:00:00Z"}},"purchased":{"balance_usd":25}}`
		case strings.HasPrefix(req.URL, accountUsagePath+"?"):
			body = `{"range":"24h","scope":"self","granularity":"hour","totals":{"request_count":3,"usage_usd":1.2,"input_tokens":100,"cached_input_tokens":0,"output_tokens":200}}`
		default:
			t.Fatalf("unexpected quota URL %q", req.URL)
		}
		return hostOK(pluginapi.HTTPResponse{StatusCode: 200, Body: []byte(body)}), nil
	}}
	m := NewManager(NewHostBridge(f.call))
	m.cfg = config.Config{BaseURL: "https://quota.test", RequestTimeout: time.Second}
	return m, f
}

func TestQuotaProviderDispatchAndCredentialContext(t *testing.T) {
	m, f := quotaProviderFixture(t)

	raw, err := m.HandleCall(pluginabi.MethodQuotaIdentifier, nil)
	if err != nil {
		t.Fatal(err)
	}
	var identifier struct {
		Identifier string `json:"identifier"`
	}
	decodeResult(t, raw, &identifier)
	if identifier.Identifier != ProviderID {
		t.Fatalf("identifier = %q", identifier.Identifier)
	}

	raw, err = m.HandleCall(pluginabi.MethodQuotaDescribe, []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	var description pluginapi.QuotaDescribeResponse
	decodeResult(t, raw, &description)
	if description.SupportsReset || len(description.SupportedProviders) != 1 || description.SupportedProviders[0] != ProviderID {
		t.Fatalf("description = %+v", description)
	}
	if description.DisplayName != "Ollama" {
		t.Fatalf("display name = %q", description.DisplayName)
	}

	req := struct {
		pluginapi.QuotaFetchRequest
		HostCallbackID string `json:"host_callback_id"`
	}{
		QuotaFetchRequest: pluginapi.QuotaFetchRequest{
			AuthIndex: "index-not-config-order", AuthID: "credential-id", Provider: ProviderID,
			Attributes:  map[string]string{"api_key": quotaProviderSecret},
			StorageJSON: []byte(`{"type":"ollama","api_key":"` + quotaProviderSecret + `"}`),
		},
		HostCallbackID: "quota-callback",
	}
	raw, err = m.HandleCall(pluginabi.MethodQuotaFetch, mustJSON(req))
	if err != nil {
		t.Fatal(err)
	}
	var got pluginapi.QuotaFetchResponse
	decodeResult(t, raw, &got)
	// Currency summary metrics: balance + purchased balance + the text usage line.
	if len(got.Summary) != 3 {
		t.Fatalf("summary = %+v", got.Summary)
	}
	if got.Summary[0].Key != "balance_usd" || got.Summary[0].Format != "currency" || math.Abs(got.Summary[0].Value-72.5) > 1e-9 {
		t.Fatalf("balance metric = %+v", got.Summary[0])
	}
	if got.Summary[1].Key != "purchased_balance_usd" || math.Abs(got.Summary[1].Value-25) > 1e-9 {
		t.Fatalf("purchased metric = %+v", got.Summary[1])
	}
	if len(got.Groups) != 1 || len(got.Groups[0].Buckets) != 1 {
		t.Fatalf("groups = %+v", got.Groups)
	}
	// included: 72.5 of 100 USD → 0.275 remaining.
	if fraction := got.Groups[0].Buckets[0].RemainingFraction; math.Abs(fraction-0.275) > 1e-9 {
		t.Fatalf("included remaining fraction = %v", fraction)
	}
	if got.Groups[0].Buckets[0].ResetTime != "2026-10-31T00:00:00Z" {
		t.Fatalf("bucket reset = %q", got.Groups[0].Buckets[0].ResetTime)
	}
	for _, call := range f.callsOf(pluginabi.MethodHostHTTPDo) {
		var wire hostHTTPReq
		if json.Unmarshal(call.payload, &wire) != nil {
			t.Fatalf("invalid callback payload")
		}
		wire.Headers.Del("Authorization")
		redacted, _ := json.Marshal(wire)
		if strings.Contains(string(redacted), quotaProviderSecret) {
			t.Fatal("credential leaked outside authorization header")
		}
	}
}

func TestQuotaProviderNativeAndLegacyNormalizationMatch(t *testing.T) {
	m, _ := quotaProviderFixture(t)
	p := newQuotaProvider(m, "quota-callback")
	got, err := p.FetchQuota(context.Background(), pluginapi.QuotaFetchRequest{
		Provider: ProviderID, StorageJSON: []byte(`{"type":"ollama","api_key":"` + quotaProviderSecret + `"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := fetchQuota(context.Background(), m.bridge.withHostCallbackID("quota-callback"), m.cfg.BaseURL, m.cfg.RequestTimeout, quotaProviderSecret)
	if err != nil {
		t.Fatal(err)
	}
	want := nativeQuotaResponse(legacy)
	if string(mustJSON(got)) != string(mustJSON(want)) {
		t.Fatalf("native and legacy normalization differ:\n got %s\nwant %s", mustJSON(got), mustJSON(want))
	}
}

func TestQuotaProviderValidationUnknownsAndReset(t *testing.T) {
	m, f := quotaProviderFixture(t)
	p := newQuotaProvider(m, "quota-callback")
	for name, req := range map[string]pluginapi.QuotaFetchRequest{
		"provider mismatch": {Provider: "other", Attributes: map[string]string{"api_key": quotaProviderSecret}, StorageJSON: []byte(`{"type":"ollama","api_key":"` + quotaProviderSecret + `"}`)},
		"attributes only":   {Provider: ProviderID, Attributes: map[string]string{"api_key": quotaProviderSecret}},
		"malformed storage": {Provider: ProviderID, StorageJSON: []byte(`{"api_key":`)},
		"other type":        {Provider: ProviderID, Attributes: map[string]string{"api_key": quotaProviderSecret}, StorageJSON: []byte(`{"type":"other","api_key":"` + quotaProviderSecret + `"}`)},
		"other provider":    {Provider: ProviderID, Attributes: map[string]string{"api_key": quotaProviderSecret}, StorageJSON: []byte(`{"provider":"other","api_key":"` + quotaProviderSecret + `"}`)},
		"missing identity":  {Provider: ProviderID, StorageJSON: []byte(`{"api_key":"` + quotaProviderSecret + `"}`)},
		"conflicting IDs":   {Provider: ProviderID, StorageJSON: []byte(`{"type":"ollama","provider":"other","api_key":"` + quotaProviderSecret + `"}`)},
		"key conflict":      {Provider: ProviderID, Attributes: map[string]string{"api_key": quotaProviderSecret}, StorageJSON: []byte(`{"type":"ollama","api_key":"different-secret"}`)},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := p.FetchQuota(context.Background(), req); err == nil || strings.Contains(err.Error(), quotaProviderSecret) {
				t.Fatalf("error = %v", err)
			}
		})
	}
	if got := len(f.callsOf(pluginabi.MethodHostHTTPDo)); got != 0 {
		t.Fatalf("invalid credentials made HTTP calls: %d", got)
	}
	before := len(f.callsOf(pluginabi.MethodHostHTTPDo))
	if _, err := p.ResetQuota(context.Background(), pluginapi.QuotaResetRequest{Provider: ProviderID, Attributes: map[string]string{"api_key": quotaProviderSecret}}); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("reset error = %v", err)
	}
	if after := len(f.callsOf(pluginabi.MethodHostHTTPDo)); after != before {
		t.Fatalf("reset made HTTP calls: before=%d after=%d", before, after)
	}

	// A snapshot without balances or windows normalizes to no groups.
	empty := nativeQuotaResponse(quotaUsage{})
	if len(empty.Groups) != 0 || len(empty.Summary) != 0 {
		t.Fatalf("empty normalization = %+v", empty)
	}

	raw, err := m.HandleCall(pluginabi.MethodQuotaFetch, []byte(`{"provider":`))
	if err != nil {
		t.Fatal(err)
	}
	var env pluginabi.Envelope
	if json.Unmarshal(raw, &env) != nil || env.OK || env.Error == nil || env.Error.Code != "invalid_request" {
		t.Fatalf("malformed envelope = %s", raw)
	}
}

func TestQuotaProviderFetchesTrustedStorageByAuthIndex(t *testing.T) {
	m, f := quotaProviderFixture(t)
	f.responder = func(method string, payload []byte) ([]byte, error) {
		if method == pluginabi.MethodHostAuthGet {
			var req struct {
				AuthIndex      string `json:"auth_index"`
				HostCallbackID string `json:"host_callback_id"`
			}
			if json.Unmarshal(payload, &req) != nil || req.AuthIndex != "trusted-index" || req.HostCallbackID != "quota-callback" {
				t.Fatalf("auth get request = %s", payload)
			}
			return hostOK(pluginapi.HostAuthGetResponse{AuthIndex: req.AuthIndex, JSON: json.RawMessage(`{"provider":"ollama","api_key":"` + quotaProviderSecret + `"}`)}), nil
		}
		if method == pluginabi.MethodHostHTTPDo {
			var req hostHTTPReq
			_ = json.Unmarshal(payload, &req)
			body := `{"included":{"balance_usd":9.5}}`
			return hostOK(pluginapi.HTTPResponse{StatusCode: 200, Body: []byte(body)}), nil
		}
		return hostOK(map[string]any{}), nil
	}
	_, err := newQuotaProvider(m, "quota-callback").FetchQuota(context.Background(), pluginapi.QuotaFetchRequest{
		Provider: ProviderID, AuthIndex: "trusted-index", Attributes: map[string]string{"api_key": quotaProviderSecret},
	})
	if err != nil {
		t.Fatal(err)
	}
}