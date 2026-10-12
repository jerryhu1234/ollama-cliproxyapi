// Predictor adapter: maps the Ollama account API (balance + usage) into
// predictor.AccountSnapshot values. Sits in the plugin package so it can
// reuse the account shapes (quota.go) and the digest identity helpers,
// while the predictor package stays provider-agnostic.
package plugin

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"

	"ollama-cliproxyapi/internal/predictor"
)

// accountGet performs one authenticated account-API GET via the host
// bridge, bounded by quotaMaxTimeout like the interactive quota path. A
// 429 is annotated with the upstream Retry-After hint so the predictor
// engine backs off exactly as long as the account surface demands.
func (m *Manager) accountGet(ctx context.Context, base, key, path string) ([]byte, error) {
	if m.bridge == nil {
		return nil, fmt.Errorf("account bridge unavailable")
	}
	ctx, cancel := context.WithTimeout(ctx, quotaMaxTimeout)
	defer cancel()
	resp, errDo := m.bridge.Do(ctx, pluginapi.HTTPRequest{
		Method: http.MethodGet,
		URL:    base + path,
		Headers: http.Header{
			"Authorization": []string{"Bearer " + key},
			"Accept":        []string{"application/json"},
		},
	})
	if errDo != nil {
		return nil, fmt.Errorf("account request failed")
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, predictor.RetryAfter(
			fmt.Errorf("account request rejected (%d)%s", resp.StatusCode, apiErrorSuffix(resp.StatusCode, resp.Body)),
			retryAfterSeconds(resp.Headers),
		)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("account request rejected (%d)%s", resp.StatusCode, apiErrorSuffix(resp.StatusCode, resp.Body))
	}
	return resp.Body, nil
}

// retryAfterFloor keeps a missing or misreported Retry-After meaningful:
// the account surface allows 10 req/min per user, so a minute of cooling
// down is the documented worst-case wait.
const retryAfterFloor = 60 * time.Second

// retryAfterSeconds parses the Retry-After header (delta-seconds form) with
// a 60s floor and a 1h ceiling so an upstream bug cannot stall sampling.
func retryAfterSeconds(h http.Header) time.Duration {
	if h == nil {
		return retryAfterFloor
	}
	raw := strings.TrimSpace(h.Get("Retry-After"))
	secs, err := strconv.Atoi(raw)
	if err != nil || secs <= 0 {
		return retryAfterFloor
	}
	if secs > 3600 {
		secs = 3600
	}
	return time.Duration(secs) * time.Second
}

// fetchPredictorSnapshot observes one credential's account state: balance
// windows plus the hourly usage shape (predictor.FetchFn). The 24h usage
// call is best-effort — balance alone still populates the window series.
func (m *Manager) fetchPredictorSnapshot(ctx context.Context, key string) (predictor.AccountSnapshot, error) {
	m.mu.RLock()
	baseURL := m.cfg.BaseURL
	m.mu.RUnlock()
	if localBaseURL(baseURL) {
		return predictor.AccountSnapshot{}, fmt.Errorf("quota API requires Ollama cloud (ollama.com); local servers expose no account surface")
	}
	base, err := AccountAPIBase(baseURL)
	if err != nil {
		return predictor.AccountSnapshot{}, err
	}
	id, label := quotaIdentity(key)
	snap := predictor.AccountSnapshot{
		KeyID: id, Label: label, At: time.Now().UTC(), PurchasedUSD: -1,
	}
	rawBalance, err := m.accountGet(ctx, base, key, accountBalancePath)
	if err != nil {
		return predictor.AccountSnapshot{}, err
	}
	var balance balanceResponse
	if err := json.Unmarshal(rawBalance, &balance); err != nil {
		return predictor.AccountSnapshot{}, fmt.Errorf("account balance response invalid: %v", err)
	}
	if balance.Included.AllowanceUSD > 0 {
		w := predictor.USDWindow{
			Remaining: balance.Included.BalanceUSD,
			Cap:       balance.Included.AllowanceUSD,
		}
		if balance.Included.Period != nil {
			w.ResetsAt = parseRFC3339Time(balance.Included.Period.Until)
		}
		snap.Included = &w
	}
	if balance.Included.Session != nil {
		snap.Session = &predictor.PctWindow{
			RemainingPercent: balance.Included.Session.RemainingPercent,
			ResetsAt:         parseRFC3339Time(balance.Included.Session.ResetsAt),
		}
	}
	if balance.Included.Weekly != nil {
		snap.Weekly = &predictor.PctWindow{
			RemainingPercent: balance.Included.Weekly.RemainingPercent,
			ResetsAt:         parseRFC3339Time(balance.Included.Weekly.ResetsAt),
		}
	}
	snap.PurchasedUSD = balance.Purchased.BalanceUSD

	// Hourly shape (best-effort): a failed or lagging usage call leaves the
	// windows fully populated and the profile flat-rate until it lands.
	rawUsage, err := m.accountGet(ctx, base, key, accountUsagePath+"?range="+usageRangeDefault+"&scope="+usageScopeDefault)
	if err == nil {
		var usageResp usageResponse
		if json.Unmarshal(rawUsage, &usageResp) == nil {
			for _, b := range usageResp.Buckets {
				from := parseRFC3339Time(b.From)
				if from.IsZero() {
					continue
				}
				snap.Hours = append(snap.Hours, predictor.HourUsage{
					HourStart: from, UsageUSD: b.UsageUSD, Partial: b.Partial,
				})
			}
		}
	}
	return snap, nil
}

// parseRFC3339Time converts an upstream ISO instant to UTC; zero when
// unparseable (the predictor treats a zero reset as "no reset bound").
func parseRFC3339Time(raw string) time.Time {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}
	}
	for _, layout := range []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05Z",
		"2006-01-02",
	} {
		if t, err := time.Parse(layout, raw); err == nil {
			return t.UTC()
		}
	}
	return time.Time{}
}
