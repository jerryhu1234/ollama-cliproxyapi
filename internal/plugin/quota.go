package plugin

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"

	"ollama-cliproxyapi/internal/config"
	"ollama-cliproxyapi/resources"
)

// Account endpoints (Ollama account surface). They live on the provider
// authority (https://ollama.com or a mirror), NOT on local Ollama servers:
// a local base-url (allow-http) exposes no balance/usage API, and those
// credentials are reported as unavailable instead of being probed.
// A provider API key authenticates them (Authorization: Bearer <key>).
// Both endpoints are rate limited to 10 requests/minute/user (HTTP 429 with
// Retry-After), so each snapshot costs exactly two upstream calls.
const (
	accountBalancePath = "/api/balance"
	accountUsagePath   = "/api/usage"
	// usageRangeDefault is the usage window rendered on the quota page; the
	// upstream also accepts 7d/30d. usageScopeDefault counts only the
	// credential's own consumption.
	usageRangeDefault = "24h"
	usageScopeDefault = "self"
	// quotaMaxTimeout bounds one account call: the management UI is
	// interactive, so a stalled upstream must fail fast rather than inherit
	// the (request-timeout, default 5m) upstream budget.
	quotaMaxTimeout = 30 * time.Second
)

// quotaRow is one metered window as rendered by the page. Rows carry a
// pre-composed human text line (currency formatting stays on the Go side),
// an optional percent/used/cap triple for the progress bar, and the reset
// instant when the upstream reports one.
type quotaRow struct {
	Name     string  `json:"name"`
	Label    string  `json:"label"`
	Status   string  `json:"status"`
	Percent  int     `json:"percent"`
	Used     float64 `json:"used,omitempty"`
	Cap      float64 `json:"cap,omitempty"`
	ResetsAt string  `json:"resets_at,omitempty"`
	Text     string  `json:"text,omitempty"`
}

// quotaMetric is one label/value fact line under the windows.
type quotaMetric struct {
	Label string `json:"label"`
	Value string `json:"value"`
}

// quotaUsage is one credential's account snapshot.
type quotaUsage struct {
	Rows    []quotaRow    `json:"rows"`
	Metrics []quotaMetric `json:"metrics,omitempty"`
	// Numeric facts for the native ABI quota renderer (QuotaMetric.Value is
	// numeric); the Metrics lines above serve the management-center page.
	BalanceUSD   float64 `json:"-"`
	PurchasedUSD float64 `json:"-"`
	// Plan labels the account when the legacy percent-window shape reported
	// it implicitly; currently unused for Ollama but kept for ABI parity.
	RefreshedAt string `json:"refreshed_at"`
}

type quotaCard struct {
	KeyID string      `json:"key_id"`
	Label string      `json:"label"`
	Usage *quotaUsage `json:"usage,omitempty"`
	Error string      `json:"error,omitempty"`
}

type quotaRequest struct {
	KeyID string `json:"key_id"`
}

type quotaList struct {
	Cards []quotaCard `json:"cards"`
}

// ---- account API response shapes (docs.ollama.com) -----------------------

// usdPeriod is the billing period the included allowance belongs to.
type usdPeriod struct {
	From  string `json:"from"`
	Until string `json:"until"`
}

// percentWindow is the LEGACY plan shape: rolling windows reported as a
// remaining percentage (0-100) plus the reset instant.
type percentWindow struct {
	RemainingPercent float64 `json:"remaining_percent"`
	ResetsAt         string  `json:"resets_at"`
}

type balanceResponse struct {
	Included struct {
		BalanceUSD   float64        `json:"balance_usd"`
		AllowanceUSD float64        `json:"allowance_usd"`
		Period       *usdPeriod     `json:"period"`
		Session      *percentWindow `json:"session"`
		Weekly       *percentWindow `json:"weekly"`
	} `json:"included"`
	Purchased struct {
		BalanceUSD float64 `json:"balance_usd"`
	} `json:"purchased"`
}

type usageTotals struct {
	RequestCount      int64   `json:"request_count"`
	UsageUSD          float64 `json:"usage_usd"`
	InputTokens       int64   `json:"input_tokens"`
	CachedInputTokens int64   `json:"cached_input_tokens"`
	OutputTokens      int64   `json:"output_tokens"`
}

type usageResponse struct {
	Range       string      `json:"range"`
	Scope       string      `json:"scope"`
	Granularity string      `json:"granularity"`
	Totals      usageTotals `json:"totals"`
}

// quotaIdentity returns the stable per-key card identity: the same
// digest-derived auth record ID the executor and materialization use.
func quotaIdentity(key string) (id, label string) {
	hash := authKeyHash(key)
	return authRecordIDFromHash(hash), "Ollama credential " + hash[:12]
}

func (m *Manager) HandleManagement(ctx context.Context, req pluginapi.ManagementRequest) (pluginapi.ManagementResponse, error) {
	if req.Method == http.MethodGet && req.Path == "/v0/resource/plugins/"+pluginName+"/quota" {
		return pluginapi.ManagementResponse{Headers: http.Header{"Content-Type": []string{"text/html; charset=utf-8"}}, Body: []byte(resources.QuotaPage)}, nil
	}
	if req.Method != http.MethodPost || req.Path != "/v0/management/plugins/"+pluginName+"/quota-usage" {
		return pluginapi.ManagementResponse{StatusCode: http.StatusNotFound, Body: []byte(`{"error":"not found"}`)}, nil
	}
	var body quotaRequest
	if len(req.Body) > 0 {
		if err := json.Unmarshal(req.Body, &body); err != nil {
			return pluginapi.ManagementResponse{StatusCode: http.StatusBadRequest, Body: []byte(`{"error":"invalid request"}`)}, nil
		}
	}
	m.mu.RLock()
	baseURL, timeout := m.cfg.BaseURL, m.cfg.RequestTimeout
	keys := append([]config.APIKey(nil), m.cfg.APIKeys...)
	m.mu.RUnlock()
	if body.KeyID == "" {
		// List is deliberately cheap: no upstream account calls. The page
		// auto-refreshes only cards it has no cached snapshot for, so a
		// reload never fan-outs one balance call per key against the
		// 10 req/min account rate limit.
		cards := make([]quotaCard, 0, len(keys))
		for _, key := range keys {
			id, label := quotaIdentity(key.Value)
			cards = append(cards, quotaCard{KeyID: id, Label: label})
		}
		return quotaJSON(quotaList{Cards: cards})
	}
	for _, key := range keys {
		id, label := quotaIdentity(key.Value)
		if id != body.KeyID {
			continue
		}
		usage, err := fetchQuota(ctx, m.bridge, baseURL, timeout, key.Value)
		if err != nil {
			// One unreachable or unsupported account must not blank the
			// whole page: the card carries the failure so the others still
			// render, and the credential hash stays as the label.
			return quotaJSON(quotaCard{KeyID: id, Label: label, Error: err.Error()})
		}
		return quotaJSON(quotaCard{KeyID: id, Label: label, Usage: &usage})
	}
	return pluginapi.ManagementResponse{StatusCode: http.StatusNotFound, Body: []byte(`{"error":"unknown quota key"}`)}, nil
}

func quotaJSON(v any) (pluginapi.ManagementResponse, error) {
	body, err := json.Marshal(v)
	if err != nil {
		return pluginapi.ManagementResponse{}, fmt.Errorf("quota response encoding failed")
	}
	return pluginapi.ManagementResponse{Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body}, nil
}

// AccountAPIBase derives the account API root from the configured base-url.
// The Ollama base-url is a bare authority (https://ollama.com or
// http://localhost:11434), so the host and any reverse-proxy path prefix is
// preserved; a trailing /v1 (OpenAI surface) is trimmed so base values
// copied from the completion endpoint still reach /api/*.
func AccountAPIBase(baseURL string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || u.Host == "" {
		return "", fmt.Errorf("account API base unavailable: base-url is not an absolute URL")
	}
	if u.Scheme != "https" && u.Scheme != "http" {
		return "", fmt.Errorf("account API base unavailable: unsupported scheme %q", u.Scheme)
	}
	path := strings.TrimSuffix(u.Path, "/")
	path = strings.TrimSuffix(path, "/v1")
	return u.Scheme + "://" + u.Host + path, nil
}

// localBaseURL reports whether the configured base points at a local Ollama
// server. Local servers expose no /api/balance or /api/usage surface, so the
// quota calls are skipped with an explanation instead of a confusing 404.
func localBaseURL(baseURL string) bool {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	return host == "localhost" || host == "127.0.0.1" || host == "::1" || strings.HasSuffix(host, ".localhost")
}

// fetchQuota reads one credential's account snapshot: the USD balance and
// window meters come from /api/balance, and the recent consumption totals
// from /api/usage. The usage call is best-effort — a snapshot with balance
// but without usage totals is still complete for the page's purposes.
func fetchQuota(ctx context.Context, bridge *HostBridge, baseURL string, timeout time.Duration, key string) (quotaUsage, error) {
	if bridge == nil {
		return quotaUsage{}, fmt.Errorf("account bridge unavailable")
	}
	if localBaseURL(baseURL) {
		return quotaUsage{}, fmt.Errorf("quota API requires Ollama cloud (ollama.com); local servers expose no account surface")
	}
	base, err := AccountAPIBase(baseURL)
	if err != nil {
		return quotaUsage{}, err
	}
	if timeout <= 0 || timeout > quotaMaxTimeout {
		timeout = quotaMaxTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	get := func(path string) ([]byte, http.Header, error) {
		resp, errDo := bridge.Do(ctx, pluginapi.HTTPRequest{
			Method: http.MethodGet,
			URL:    base + path,
			Headers: http.Header{
				"Authorization": []string{"Bearer " + key},
				"Accept":        []string{"application/json"},
			},
		})
		if errDo != nil {
			return nil, nil, fmt.Errorf("account request failed")
		}
		if resp.StatusCode != http.StatusOK {
			return nil, resp.Headers, fmt.Errorf("account request rejected (%d)%s", resp.StatusCode, apiErrorSuffix(resp.StatusCode, resp.Body))
		}
		return resp.Body, resp.Headers, nil
	}

	rawBalance, _, err := get(accountBalancePath)
	if err != nil {
		return quotaUsage{}, err
	}
	var balance balanceResponse
	if err := json.Unmarshal(rawBalance, &balance); err != nil {
		// Carry the decoder's own reason: it names the offending field, which
		// is the difference between a one-minute and a one-hour diagnosis.
		return quotaUsage{}, fmt.Errorf("account balance response invalid: %v", err)
	}
	usage := quotaUsage{RefreshedAt: time.Now().UTC().Format(time.RFC3339)}
	usage.Rows, usage.Metrics, usage.BalanceUSD, usage.PurchasedUSD = balanceRows(balance)

	// Usage totals (best-effort): a wrong range/scope parameter is rejected
	// by the upstream with 400, and rate limiting here must never blank the
	// balance card.
	rawUsage, _, err := get(accountUsagePath + "?range=" + usageRangeDefault + "&scope=" + usageScopeDefault)
	if err == nil {
		var usageResp usageResponse
		if json.Unmarshal(rawUsage, &usageResp) == nil {
			usage.Metrics = append(usage.Metrics, usageTotalsMetric(usageResp.Totals))
		}
	}
	return usage, nil
}

// balanceRows converts a decoded balance snapshot into the page's ordered
// rows and headline metrics. Shape handling:
//   - Current (USD) shape: allowance/balance/period present → an "Included
//     balance" meter row plus balance/purchased metrics.
//   - Legacy plan shape: included.session / included.weekly windows report
//     a remaining percentage; they render as percent-only rows with resets.
func balanceRows(b balanceResponse) (rows []quotaRow, metrics []quotaMetric, balanceUSD, purchasedUSD float64) {
	included := b.Included
	if included.AllowanceUSD > 0 {
		row := quotaRow{
			Name:   "included",
			Label:  "Included balance",
			Status: "ok",
			Cap:    included.AllowanceUSD,
		}
		remaining := included.BalanceUSD
		used := math.Max(0, included.AllowanceUSD-remaining)
		row.Used = used
		row.Percent = clampPercent(used / included.AllowanceUSD * 100)
		if remaining <= 0 {
			row.Status = "exceeded"
		}
		if included.Period != nil {
			row.ResetsAt = periodRFC3339(included.Period.Until)
		}
		row.Text = fmt.Sprintf("$%.2f left of $%.2f", remaining, included.AllowanceUSD)
		rows = append(rows, row)
	}
	metrics = append(metrics, quotaMetric{
		Label: "Balance",
		Value: fmt.Sprintf("$%.2f included", included.BalanceUSD),
	})
	if purchased := b.Purchased.BalanceUSD; purchased > 0 {
		metrics = append(metrics, quotaMetric{Label: "Purchased balance", Value: fmt.Sprintf("$%.2f", purchased)})
	}
	rows = appendPercentRow(rows, "session", "Session window", included.Session)
	rows = appendPercentRow(rows, "weekly", "Weekly window", included.Weekly)
	return rows, metrics, included.BalanceUSD, b.Purchased.BalanceUSD
}

// appendPercentRow renders one legacy remaining-percentage window.
func appendPercentRow(rows []quotaRow, name, label string, w *percentWindow) []quotaRow {
	if w == nil {
		return rows
	}
	remaining := math.Max(0, math.Min(100, w.RemainingPercent))
	usedPercent := int(math.Round(100 - remaining))
	row := quotaRow{
		Name:     name,
		Label:    label,
		Status:   "ok",
		Percent:  usedPercent,
		ResetsAt: periodRFC3339(w.ResetsAt),
	}
	if remaining <= 0 {
		row.Status = "exceeded"
	}
	row.Text = fmt.Sprintf("%d%% remaining", int(math.Round(remaining)))
	return append(rows, row)
}

// usageTotalsMetric renders the 24h consumption totals as one fact line.
func usageTotalsMetric(t usageTotals) quotaMetric {
	parts := []string{fmt.Sprintf("%d requests", t.RequestCount), "$" + strconv.FormatFloat(t.UsageUSD, 'f', 2, 64)}
	if extra := tokenParts(t); extra != "" {
		parts = append(parts, extra)
	}
	return quotaMetric{Label: "Usage (24h)", Value: strings.Join(parts, " · ")}
}

// tokenParts folds the token counters into one compact token fragment,
// skipping cached when zero so typical responses stay short.
func tokenParts(t usageTotals) string {
	if t.InputTokens == 0 && t.OutputTokens == 0 {
		return ""
	}
	fragments := []string{formatTokens(t.InputTokens) + " in"}
	if t.CachedInputTokens > 0 {
		fragments = append(fragments, formatTokens(t.CachedInputTokens)+" cached")
	}
	fragments = append(fragments, formatTokens(t.OutputTokens)+" out")
	return strings.Join(fragments, " / ")
}

// formatTokens abbreviates large counters the way usage consoles do.
func formatTokens(n int64) string {
	switch {
	case n >= 1_000_000:
		return strconv.FormatFloat(float64(n)/1_000_000, 'f', 1, 64) + "M"
	case n >= 1_000:
		return strconv.FormatFloat(float64(n)/1_000, 'f', 1, 64) + "k"
	default:
		return strconv.FormatInt(n, 10)
	}
}

// clampPercent bounds a computed percent to the bar's 0-100 range.
func clampPercent(p float64) int {
	return int(math.Max(0, math.Min(100, p)))
}

// periodRFC3339 turns an upstream ISO instant into the RFC3339 UTC form the
// page's reset renderer understands; anything unparseable is dropped so the
// page does not invent a date.
func periodRFC3339(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	for _, layout := range []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05Z",
		"2006-01-02T15:04:05",
		"2006-01-02",
	} {
		if t, err := time.Parse(layout, raw); err == nil {
			return t.UTC().Format(time.RFC3339)
		}
	}
	return ""
}

// apiErrorSuffix extracts the upstream's `{"error":"..."}` diagnostic for a
// non-2xx account response, appending rate-limit guidance when present. The
// raw body never reaches the page — only the short vendor message.
func apiErrorSuffix(status int, body []byte) string {
	if len(body) > 0 {
		var payload struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(body, &payload) == nil && strings.TrimSpace(payload.Error) != "" {
			return ": " + strings.TrimSpace(payload.Error)
		}
	}
	if status == http.StatusTooManyRequests {
		return ": rate limited, retry shortly"
	}
	return ""
}
