package plugin

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginabi"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"
)

func TestAuthProviderParseAndRefresh(t *testing.T) {
	secret := "sk-auth-secret-1"
	p := authProvider{}
	rawBytes := []byte(`{"type":"ollama","id":"stable","label":"Primary","api_key":"` + secret + `"}`)
	parsed, err := p.ParseAuth(context.Background(), pluginapi.AuthParseRequest{
		Provider: ProviderID, FileName: "key.json",
		RawJSON: rawBytes,
	})
	if err != nil || !parsed.Handled || parsed.Auth.ID != "stable" || parsed.Auth.Label != "Primary" || parsed.Auth.Attributes["api_key"] != secret || string(parsed.Auth.StorageJSON) != string(rawBytes) {
		t.Fatalf("parsed auth = %#v, err=%v", parsed, err)
	}
	if _, err := p.ParseAuth(context.Background(), pluginapi.AuthParseRequest{RawJSON: []byte(`{"type":"other","api_key":"x1"}`)}); err != nil {
		t.Fatalf("unrelated auth error = %v", err)
	}
	invalid, err := p.ParseAuth(context.Background(), pluginapi.AuthParseRequest{Provider: ProviderID, RawJSON: []byte(`{"type":"ollama"}`)})
	if err == nil || invalid.Handled || strings.Contains(err.Error(), secret) {
		t.Fatalf("invalid auth = %#v, err=%v", invalid, err)
	}
	refreshed, err := p.RefreshAuth(context.Background(), pluginapi.AuthRefreshRequest{AuthID: "stable", AuthProvider: ProviderID, Attributes: parsed.Auth.Attributes})
	if err != nil || refreshed.Auth.Attributes["api_key"] != secret {
		t.Fatalf("refresh = %#v, err=%v", refreshed, err)
	}
}

func TestAuthProviderAliasesAndFilenameFallback(t *testing.T) {
	p := authProvider{}
	parsed, err := p.ParseAuth(context.Background(), pluginapi.AuthParseRequest{
		FileName: "custom-file.json",
		RawJSON:  []byte(`{"provider":"ollama","api_key":"sk-alias-key"}`),
	})
	if err != nil || !parsed.Handled || parsed.Auth.ID != "custom-file.json" || parsed.Auth.Attributes["api_key"] != "sk-alias-key" {
		t.Fatalf("alias parse = %#v, err=%v", parsed, err)
	}

	unhandled, err := p.ParseAuth(context.Background(), pluginapi.AuthParseRequest{
		FileName: "other.json",
		RawJSON:  []byte(`{"provider":"other-provider","api_key":"sk-other-key"}`),
	})
	if err != nil || unhandled.Handled {
		t.Fatalf("unhandled parse = %#v, err=%v", unhandled, err)
	}
}

func TestAuthProviderFileBackedDefersCanonicalIDToHost(t *testing.T) {
	p := authProvider{}
	raw := []byte(`{"type":"ollama","id":"ollama-key-H","api_key":"sk-file-key","email":"file@example.test"}`)
	parsed, err := p.ParseAuth(context.Background(), pluginapi.AuthParseRequest{
		Provider: ProviderID,
		Path:     "/auth/ollama-key-H.json",
		FileName: "ollama-key-H.json",
		RawJSON:  raw,
	})
	if err != nil || !parsed.Handled {
		t.Fatalf("file-backed parse = %#v, err=%v", parsed, err)
	}
	if parsed.Auth.ID != "" {
		t.Fatalf("file-backed ID = %q, want empty for host path canonicalization", parsed.Auth.ID)
	}
	if parsed.Auth.FileName != "ollama-key-H.json" || parsed.Auth.Attributes["api_key"] != "sk-file-key" || parsed.Auth.Metadata["email"] != "file@example.test" || string(parsed.Auth.StorageJSON) != string(raw) {
		t.Fatalf("file-backed fields changed: %#v", parsed.Auth)
	}
}

func TestAuthProviderNonFileKeepsLegacyIdentity(t *testing.T) {
	p := authProvider{}
	for _, tc := range []struct {
		name string
		raw  string
		want string
	}{
		{name: "json id", raw: `{"type":"ollama","id":"stable-json-id","api_key":"sk-memory-key"}`, want: "stable-json-id"},
		{name: "filename fallback", raw: `{"type":"ollama","api_key":"sk-memory-key"}`, want: "memory.json"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parsed, err := p.ParseAuth(context.Background(), pluginapi.AuthParseRequest{FileName: "memory.json", RawJSON: []byte(tc.raw)})
			if err != nil || !parsed.Handled || parsed.Auth.ID != tc.want {
				t.Fatalf("non-file parse = %#v, err=%v, want ID %q", parsed, err, tc.want)
			}
		})
	}
}

func TestAuthProviderMalformedRecognizedInput(t *testing.T) {

	p := authProvider{}
	parsed, err := p.ParseAuth(context.Background(), pluginapi.AuthParseRequest{
		Provider: ProviderID,
		RawJSON:  []byte(`{"type":"ollama","api_key":"sk-malformed-secret-1"`),
	})
	if err == nil || parsed.Handled || strings.Contains(err.Error(), "sk-malformed-secret-1") {
		t.Fatalf("malformed recognized auth = %#v, err=%v", parsed, err)
	}
	if err.Error() != "ollama auth record has invalid JSON" {
		t.Fatalf("malformed recognized auth error = %v", err)
	}

	unrelated, err := p.ParseAuth(context.Background(), pluginapi.AuthParseRequest{
		RawJSON: []byte(`{"type":"ollama","api_key":"sk-malformed-secret-2"`),
	})
	if err != nil || unrelated.Handled {
		t.Fatalf("malformed unrelated auth = %#v, err=%v", unrelated, err)
	}
}

func TestAuthDispatchMethods(t *testing.T) {
	m := NewManager(nil)
	var got pluginapi.AuthParseResponse
	raw, err := m.HandleCall(pluginabi.MethodAuthIdentifier, nil)
	if err != nil || !bytesContain(raw, []byte(ProviderID)) {
		t.Fatalf("identifier = %s, err=%v", raw, err)
	}
	raw, err = m.HandleCall(pluginabi.MethodAuthParse, mustJSON(pluginapi.AuthParseRequest{Provider: ProviderID, RawJSON: []byte(`{"type":"ollama","api_key":"sk-dispatch-1"}`)}))
	if err != nil {
		t.Fatal(err)
	}
	decodeResult(t, raw, &got)
	if !got.Handled || got.Auth.Attributes["api_key"] != "sk-dispatch-1" {
		t.Fatalf("parse dispatch = %#v", got)
	}
	for _, method := range []string{pluginabi.MethodAuthLoginStart, pluginabi.MethodAuthLoginPoll} {
		raw, _ = m.HandleCall(method, []byte(`{}`))
		if env := decodeEnv(t, raw); env.OK || env.Error == nil || env.Error.Code != "unsupported" {
			t.Fatalf("%s = %#v", method, env.Error)
		}
	}
	raw, err = m.HandleCall(pluginabi.MethodAuthRefresh, mustJSON(struct{ pluginapi.AuthRefreshRequest }{pluginapi.AuthRefreshRequest{AuthID: "id", AuthProvider: ProviderID, Attributes: map[string]string{"api_key": "sk-refresh-1"}}}))
	if err != nil {
		t.Fatal(err)
	}
	var refreshed pluginapi.AuthRefreshResponse
	decodeResult(t, raw, &refreshed)
	if refreshed.Auth.ID != "id" || refreshed.Auth.Attributes["api_key"] != "sk-refresh-1" {
		t.Fatalf("refresh dispatch = %#v", refreshed)
	}
}

func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }
func bytesContain(haystack, needle []byte) bool {
	return strings.Contains(string(haystack), string(needle))
}
