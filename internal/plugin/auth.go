package plugin

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"
)

type authProvider struct{}

var _ pluginapi.AuthProvider = authProvider{}

func (authProvider) Identifier() string { return ProviderID }

func (authProvider) ParseAuth(_ context.Context, req pluginapi.AuthParseRequest) (pluginapi.AuthParseResponse, error) {
	debugTrace("auth parse request provider=%s file=%s raw_bytes=%d", req.Provider, req.FileName, len(req.RawJSON))
	var raw struct {
		Type     string `json:"type"`
		Provider string `json:"provider"`
		ID       string `json:"id"`
		Label    string `json:"label"`
		APIKey   string `json:"api_key"`
		Email    string `json:"email"`
	}
	if err := json.Unmarshal(req.RawJSON, &raw); err != nil {
		if req.Provider == ProviderID {
			return pluginapi.AuthParseResponse{}, fmt.Errorf("ollama auth record has invalid JSON")
		}
		return pluginapi.AuthParseResponse{}, nil
	}
	if (req.Provider != "" && req.Provider != ProviderID) || (raw.Type != ProviderID && raw.Provider != ProviderID) {
		return pluginapi.AuthParseResponse{Handled: false}, nil
	}
	if strings.TrimSpace(raw.APIKey) == "" {
		return pluginapi.AuthParseResponse{}, fmt.Errorf("ollama auth record has no api key")
	}
	// File-backed auth identity belongs to the host: leaving ID empty lets CPA
	// canonicalize req.Path relative to auth-dir (including the .json suffix).
	// host.auth.save uses that same path-derived ID. Supplying the JSON id here
	// would make the initial save and the watcher re-parse look like two auths.
	// Non-file callers have no path to canonicalize, so retain the historical
	// JSON id / FileName fallback for compatibility.
	authID := raw.ID
	if strings.TrimSpace(req.Path) != "" {
		authID = ""
	} else if authID == "" {
		raw.ID = req.FileName
		authID = raw.ID
	}
	debugTrace("auth parse handled provider=%s file=%s id=%s api_key_present=%t api_key_length=%d email_present=%t", req.Provider, req.FileName, authID, strings.TrimSpace(raw.APIKey) != "", len(raw.APIKey), strings.TrimSpace(raw.Email) != "")
	auth := pluginapi.AuthData{
		Provider: ProviderID, ID: authID, FileName: req.FileName, Label: raw.Label, StorageJSON: req.RawJSON,
		Attributes: map[string]string{"api_key": raw.APIKey},
	}
	// The host derives a credential's email from Metadata["email"], and the
	// management panel titles the auth entry with that email. Only the bare
	// mailbox is surfaced: an absent email stays absent instead of becoming an
	// empty value the panel would have to render around.
	if email := strings.TrimSpace(raw.Email); email != "" {
		auth.Metadata = map[string]any{"email": email}
	}
	return pluginapi.AuthParseResponse{Handled: true, Auth: auth}, nil
}

func (authProvider) StartLogin(context.Context, pluginapi.AuthLoginStartRequest) (pluginapi.AuthLoginStartResponse, error) {
	return pluginapi.AuthLoginStartResponse{}, fmt.Errorf("ollama login is unsupported; configure a manual api key")
}

func (authProvider) PollLogin(context.Context, pluginapi.AuthLoginPollRequest) (pluginapi.AuthLoginPollResponse, error) {
	return pluginapi.AuthLoginPollResponse{}, fmt.Errorf("ollama login is unsupported; configure a manual api key")
}

func (authProvider) RefreshAuth(_ context.Context, req pluginapi.AuthRefreshRequest) (pluginapi.AuthRefreshResponse, error) {
	debugTrace("auth refresh request provider=%s id=%s storage_json_bytes=%d attr_names=%v metadata_names=%v", req.AuthProvider, req.AuthID, len(req.StorageJSON), mapKeys(req.Attributes), mapKeys(req.Metadata))
	return pluginapi.AuthRefreshResponse{Auth: pluginapi.AuthData{Provider: req.AuthProvider, ID: req.AuthID, StorageJSON: req.StorageJSON, Metadata: req.Metadata, Attributes: req.Attributes}}, nil
}

func mapKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	return keys
}
