// Package config loads and validates the ollama plugin configuration
// per spec 04 (configuration) and spec 05 §2 (HTTPS/timeouts).
package config

import (
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Defaults (spec 04 §2/§4).
const (
	DefaultBaseURL                = "https://ollama.com" // Ollama cloud; use http://localhost:11434 with allow-http for local
	DefaultPlaceholderAPIKey      = "ollama"             // accepted literal key for local servers that ignore auth
	DefaultModelPrefix            = "ollama"
	DefaultRefreshInterval        = 15 * time.Minute
	DefaultRequestTimeout         = 5 * time.Minute
	DefaultStreamFirstDataTimeout = 60 * time.Second
	DefaultStreamIdleTimeout      = 3 * time.Minute
	DefaultStreamTotalTimeout     = 30 * time.Minute
	DefaultStreamFinishGrace      = 15 * time.Second
	DefaultMaxResponseBytes       = int64(67108864) // 64 MiB
	DefaultResponsesCompatibility = "cpa"

	// Usage predictor defaults (docs/usage-prediction-plan.md §7).
	DefaultPredictorSampleInterval = 10 * time.Minute
	DefaultPredictorBudget         = 8 // account-API calls per minute, global
	DefaultPredictorHistory        = 7 * 24 * time.Hour
	DefaultPredictorHorizon        = 48 * time.Hour
	PredictorDataDirOff            = "off" // explicit memory-only marker
	DefaultPredictorDataDirSuffix  = ".cli-proxy-api/usage-predictor"
)

// Predictor configures the usage prediction sampler (plan §7): period
// balance/usage sampling into a local time-series store, from which the
// prediction page reads — never the upstream account API directly.
type Predictor struct {
	Enabled         bool
	Persistence     bool
	SampleInterval  time.Duration
	BudgetPerMinute int
	History         time.Duration
	Horizon         time.Duration
	DataDir         string // "" = auto-resolve; "off" = memory-only; else explicit path
	// PeerPlugins pins the sibling plugin IDs whose /prediction endpoints
	// the page may aggregate (plan §8 P3b,备选显式列表). Empty ⇒ page-side
	// auto-discovery via the management plugin list.
	PeerPlugins []string
}

type ModelPrefix struct {
	Enabled bool
	Value   string
}

type APIKey struct {
	Value string
}

type Catalog struct {
	RefreshInterval       time.Duration
	StaleWhileUnavailable bool
}

type Protocols struct {
	ChatCompletions bool
	Messages        bool
	Responses       bool
}

type RouteOverride struct {
	Protocol string `yaml:"protocol"`
	Endpoint string `yaml:"endpoint"`
}

type Config struct {
	BaseURL                string
	CatalogURL             string
	ModelPrefix            ModelPrefix
	APIKeys                []APIKey
	Catalog                Catalog
	Protocols              Protocols
	Predictor              Predictor
	RouteOverrides         map[string]RouteOverride
	AllowHTTP              bool
	RequestTimeout         time.Duration
	StreamFirstDataTimeout time.Duration
	StreamIdleTimeout      time.Duration
	StreamTotalTimeout     time.Duration
	StreamFinishGrace      time.Duration
	MaxResponseBytes       int64
	ResponsesCompatibility string
}

// rawConfig mirrors the YAML shape; pointer fields distinguish "unset"
// (apply default) from explicitly-set values including "" (validate as-is).
// Unknown fields are ignored (host may pass extra keys).
type rawConfig struct {
	BaseURL                *string                  `yaml:"base-url"`
	CatalogURL             *string                  `yaml:"catalog-url"`
	ModelPrefix            rawPrefix                `yaml:"model-prefix"`
	APIKeys                []rawKey                 `yaml:"api-keys"`
	Catalog                rawCatalog               `yaml:"catalog"`
	Protocols              rawProtocols             `yaml:"protocols"`
	Predictor              rawPredictor             `yaml:"predictor"`
	RouteOverrides         map[string]RouteOverride `yaml:"route-overrides"`
	AllowHTTP              bool                     `yaml:"allow-http"`
	RequestTimeout         *string                  `yaml:"request-timeout"`
	StreamFirstDataTimeout *string                  `yaml:"stream-first-data-timeout"`
	StreamIdleTimeout      *string                  `yaml:"stream-idle-timeout"`
	StreamTotalTimeout     *string                  `yaml:"stream-total-timeout"`
	StreamFinishGrace      *string                  `yaml:"stream-finish-grace"`
	MaxResponseBytes       *int64                   `yaml:"max-response-bytes"`
	ResponsesCompatibility *string                  `yaml:"responses-compatibility"`
}

type rawPrefix struct {
	Enabled *bool   `yaml:"enabled"`
	Value   *string `yaml:"value"`
}

type rawKey struct {
	Value string `yaml:"value"`
}

type rawCatalog struct {
	RefreshInterval       *string `yaml:"refresh-interval"`
	StaleWhileUnavailable *bool   `yaml:"stale-while-unavailable"`
}

type rawProtocols struct {
	ChatCompletions *bool `yaml:"chat-completions"`
	Messages        *bool `yaml:"messages"`
	Responses       *bool `yaml:"responses"`
}

// rawPredictor mirrors the predictor YAML block; pointers distinguish
// unset (apply default) from explicit values.
type rawPredictor struct {
	Enabled         *bool     `yaml:"enabled"`
	Persistence     *bool     `yaml:"persistence"`
	SampleInterval  *string   `yaml:"sample-interval"`
	BudgetPerMinute *int      `yaml:"budget-per-minute"`
	History         *string   `yaml:"history"`
	Horizon         *string   `yaml:"horizon"`
	DataDir         *string   `yaml:"data-dir"`
	PeerPlugins     *[]string `yaml:"peer-plugins"`
}

var (
	validProtocols = map[string]bool{"chat-completions": true, "messages": true, "responses": true}
)

// Load decodes YAML, expands ${VAR} references in api-key values only,
// applies defaults, and validates (spec 04 §6). Decode errors never echo
// decoded node values — a malformed entry (e.g. a bare-scalar API key)
// must not leak into the invalid_config envelope the host logs.
func Load(yamlBytes []byte) (Config, error) {
	var raw rawConfig
	if err := yaml.Unmarshal(yamlBytes, &raw); err != nil {
		if n := regexp.MustCompile(`line (\d+)`).FindStringSubmatch(err.Error()); n != nil {
			return Config{}, fmt.Errorf("decode config: invalid YAML structure near line %s", n[1])
		}
		return Config{}, fmt.Errorf("decode config: invalid YAML structure")
	}
	keys := make([]APIKey, len(raw.APIKeys))
	for i, k := range raw.APIKeys {
		keys[i] = APIKey{Value: os.ExpandEnv(k.Value)}
	}
	refreshInterval, err := parseDuration("catalog.refresh-interval", raw.Catalog.RefreshInterval, DefaultRefreshInterval)
	if err != nil {
		return Config{}, err
	}
	requestTimeout, err := parseDuration("request-timeout", raw.RequestTimeout, DefaultRequestTimeout)
	if err != nil {
		return Config{}, err
	}
	if requestTimeout <= 0 {
		return Config{}, fmt.Errorf("request-timeout: must be positive")
	}
	streamFirstDataTimeout, err := parseDuration("stream-first-data-timeout", raw.StreamFirstDataTimeout, DefaultStreamFirstDataTimeout)
	if err != nil {
		return Config{}, err
	}
	if streamFirstDataTimeout <= 0 {
		return Config{}, fmt.Errorf("stream-first-data-timeout: must be positive")
	}
	streamIdleTimeout, err := parseDuration("stream-idle-timeout", raw.StreamIdleTimeout, DefaultStreamIdleTimeout)
	if err != nil {
		return Config{}, err
	}
	if streamIdleTimeout <= 0 {
		return Config{}, fmt.Errorf("stream-idle-timeout: must be positive")
	}
	streamTotalTimeout, err := parseDuration("stream-total-timeout", raw.StreamTotalTimeout, DefaultStreamTotalTimeout)
	if err != nil {
		return Config{}, err
	}
	if streamTotalTimeout <= 0 {
		return Config{}, fmt.Errorf("stream-total-timeout: must be positive")
	}
	streamFinishGrace, err := parseDuration("stream-finish-grace", raw.StreamFinishGrace, DefaultStreamFinishGrace)
	if err != nil {
		return Config{}, err
	}
	if streamFinishGrace <= 0 {
		return Config{}, fmt.Errorf("stream-finish-grace: must be positive")
	}
	if streamTotalTimeout < streamFirstDataTimeout {
		return Config{}, fmt.Errorf("stream-total-timeout: must be at least stream-first-data-timeout")
	}
	if streamTotalTimeout < streamIdleTimeout {
		return Config{}, fmt.Errorf("stream-total-timeout: must be at least stream-idle-timeout")
	}
	if streamFinishGrace > streamIdleTimeout {
		return Config{}, fmt.Errorf("stream-finish-grace: must not exceed stream-idle-timeout")
	}
	predictorCfg, err := loadPredictor(&raw.Predictor)
	if err != nil {
		return Config{}, err
	}
	c := Config{
		BaseURL: orDefault(raw.BaseURL, DefaultBaseURL),
		ModelPrefix: ModelPrefix{
			Enabled: orDefault(raw.ModelPrefix.Enabled, true),
			Value:   orDefault(raw.ModelPrefix.Value, DefaultModelPrefix),
		},
		APIKeys: keys,
		Catalog: Catalog{
			RefreshInterval:       refreshInterval,
			StaleWhileUnavailable: orDefault(raw.Catalog.StaleWhileUnavailable, true),
		},
		Protocols: Protocols{
			ChatCompletions: orDefault(raw.Protocols.ChatCompletions, true),
			Messages:        orDefault(raw.Protocols.Messages, true),
			Responses:       orDefault(raw.Protocols.Responses, true),
		},
		Predictor:              predictorCfg,
		RouteOverrides:         raw.RouteOverrides,
		AllowHTTP:              raw.AllowHTTP,
		RequestTimeout:         requestTimeout,
		StreamFirstDataTimeout: streamFirstDataTimeout,
		StreamIdleTimeout:      streamIdleTimeout,
		StreamTotalTimeout:     streamTotalTimeout,
		StreamFinishGrace:      streamFinishGrace,
		MaxResponseBytes:       orDefault(raw.MaxResponseBytes, DefaultMaxResponseBytes),
		ResponsesCompatibility: strings.ToLower(strings.TrimSpace(orDefault(raw.ResponsesCompatibility, DefaultResponsesCompatibility))),
	}
	if raw.CatalogURL != nil {
		// Mirror the derived-default trim so an explicit trailing-slash
		// catalog-url cannot double up separators downstream.
		c.CatalogURL = strings.TrimRight(*raw.CatalogURL, "/")
	} else {
		// Ollama serves the OpenAI-compatible catalog at {base}/v1/models.
		// A base pasted from the OpenAI surface already ends in /v1 — trim
		// it first (same normalization as AccountAPIBase) so the derived
		// URL never doubles the segment.
		c.CatalogURL = strings.TrimSuffix(strings.TrimRight(c.BaseURL, "/"), "/v1") + "/v1/models"
	}
	if err := c.validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}

// PublicID returns the client-facing model ID (FR-003), e.g. "ollama/gpt-oss:20b".
func PublicID(c Config, upstreamID string) string {
	if c.ModelPrefix.Enabled {
		return c.ModelPrefix.Value + "/" + upstreamID
	}
	return upstreamID
}

func (c Config) validate() error {
	if err := validatePredictor(c.Predictor); err != nil {
		return err
	}
	if err := validateURL("base-url", c.BaseURL, c.AllowHTTP); err != nil {
		return err
	}
	if err := validateURL("catalog-url", c.CatalogURL, c.AllowHTTP); err != nil {
		return err
	}
	if len(c.APIKeys) == 0 {
		return fmt.Errorf("api-keys: at least one key is required")
	}
	for i, k := range c.APIKeys {
		if k.Value == "" {
			return fmt.Errorf("api-keys[%d].value: expanded to empty", i)
		}
	}
	seen := make(map[string]bool, len(c.APIKeys))
	for _, k := range c.APIKeys {
		if seen[k.Value] {
			return fmt.Errorf("api-keys: duplicate key values are not allowed")
		}
		seen[k.Value] = true
	}
	if c.Catalog.RefreshInterval < time.Minute {
		return fmt.Errorf("catalog.refresh-interval: must be at least 1m")
	}
	if c.MaxResponseBytes <= 0 {
		return fmt.Errorf("max-response-bytes: must be positive")
	}
	if c.ResponsesCompatibility != "cpa" && c.ResponsesCompatibility != "strict" {
		return fmt.Errorf("responses-compatibility: must be cpa or strict")
	}
	for name, o := range c.RouteOverrides {
		if !validProtocols[o.Protocol] {
			return fmt.Errorf("route-overrides[%s].protocol: unsupported protocol %q", name, o.Protocol)
		}
		if o.Endpoint == "" {
			return fmt.Errorf("route-overrides[%s].endpoint: must not be empty", name)
		}
		if !strings.HasPrefix(o.Endpoint, "/") {
			return fmt.Errorf("route-overrides[%s].endpoint: must start with /", name)
		}
	}
	if c.ModelPrefix.Enabled && !validPrefix(c.ModelPrefix.Value) {
		return fmt.Errorf("model-prefix.value: invalid provider-ID characters %q", c.ModelPrefix.Value)
	}
	return nil
}

func validateURL(name, raw string, allowHTTP bool) error {
	if raw == "" {
		return fmt.Errorf("%s: must not be empty", name)
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("%s: invalid URL", name)
	}
	if u.Scheme == "" || u.Host == "" {
		// Echo only scheme://host — the configured string may embed
		// userinfo credentials (https://user:key@host).
		return fmt.Errorf("%s: invalid URL %s://%s", name, u.Scheme, u.Host)
	}
	if u.RawQuery != "" || u.Fragment != "" || u.User != nil {
		return fmt.Errorf("%s: must not contain query, fragment, or userinfo", name)
	}
	// Scheme allowlist (spec 05 §2): https always; http only behind
	// allow-http; anything else (ftp://, file://, custom schemes) is
	// rejected at load instead of failing at transport time.
	switch u.Scheme {
	case "https":
	case "http":
		if !allowHTTP {
			return fmt.Errorf("%s: http scheme requires allow-http", name)
		}
	default:
		return fmt.Errorf("%s: unsupported scheme %q; use https", name, u.Scheme)
	}
	return nil
}

func validPrefix(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		alnum := r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9'
		if i == 0 && !alnum {
			return false
		}
		if !alnum && r != '.' && r != '_' && r != '-' {
			return false
		}
	}
	return true
}

// loadPredictor applies the predictor block defaults (plan §7: enabled +
// persistence on, 10m sampling, 8/min budget, 7d history, 48h horizon,
// auto data-dir unless explicitly "off").
func loadPredictor(raw *rawPredictor) (Predictor, error) {
	sampleInterval, err := parseDuration("predictor.sample-interval", raw.SampleInterval, DefaultPredictorSampleInterval)
	if err != nil {
		return Predictor{}, err
	}
	history, err := parseDuration("predictor.history", raw.History, DefaultPredictorHistory)
	if err != nil {
		return Predictor{}, err
	}
	horizon, err := parseDuration("predictor.horizon", raw.Horizon, DefaultPredictorHorizon)
	if err != nil {
		return Predictor{}, err
	}
	p := Predictor{
		Enabled:         orDefault(raw.Enabled, true),
		Persistence:     orDefault(raw.Persistence, true),
		SampleInterval:  sampleInterval,
		BudgetPerMinute: orDefault(raw.BudgetPerMinute, DefaultPredictorBudget),
		History:         history,
		Horizon:         horizon,
	}
	if raw.DataDir != nil {
		p.DataDir = strings.TrimSpace(*raw.DataDir)
	}
	if raw.PeerPlugins != nil {
		p.PeerPlugins = normalizePeerPlugins(*raw.PeerPlugins)
	}
	return p, nil
}

// normalizePeerPlugins trims, drops empties and dedupes the explicit
// sibling list, preserving first-seen order (plan §8 P3b).
func normalizePeerPlugins(in []string) []string {
	out := make([]string, 0, len(in))
	seen := make(map[string]bool, len(in))
	for _, id := range in {
		id = strings.TrimSpace(id)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		out = append(out, id)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func validatePredictor(p Predictor) error {
	if !p.Enabled {
		return nil
	}
	if p.SampleInterval < 2*time.Minute {
		return fmt.Errorf("predictor.sample-interval: must be at least 2m")
	}
	if p.BudgetPerMinute < 1 || p.BudgetPerMinute > 10 {
		return fmt.Errorf("predictor.budget-per-minute: must be 1..10 (account API allows 10/min per user)")
	}
	if p.History < 24*time.Hour {
		return fmt.Errorf("predictor.history: must be at least 24h")
	}
	if p.Horizon < time.Hour {
		return fmt.Errorf("predictor.horizon: must be at least 1h")
	}
	if p.Horizon > p.History {
		return fmt.Errorf("predictor.horizon: must not exceed predictor.history")
	}
	return nil
}

func parseDuration(name string, p *string, def time.Duration) (time.Duration, error) {
	if p == nil {
		return def, nil
	}
	d, err := time.ParseDuration(*p)
	if err != nil {
		return 0, fmt.Errorf("%s: invalid duration", name)
	}
	return d, nil
}

func orDefault[T any](p *T, def T) T {
	if p != nil {
		return *p
	}
	return def
}
