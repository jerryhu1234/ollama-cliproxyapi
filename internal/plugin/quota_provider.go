package plugin

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"
)

type quotaProvider struct {
	manager        *Manager
	hostCallbackID string
}

var _ pluginapi.QuotaProvider = quotaProvider{}

func newQuotaProvider(manager *Manager, hostCallbackID string) quotaProvider {
	return quotaProvider{manager: manager, hostCallbackID: hostCallbackID}
}

func (quotaProvider) Identifier() string { return ProviderID }

func (quotaProvider) DescribeQuota(context.Context, pluginapi.QuotaDescribeRequest) (pluginapi.QuotaDescribeResponse, error) {
	return pluginapi.QuotaDescribeResponse{
		SupportedProviders: []string{ProviderID},
		DisplayName:        "Ollama",
		SupportsReset:      false,
	}, nil
}

func (p quotaProvider) FetchQuota(ctx context.Context, req pluginapi.QuotaFetchRequest) (pluginapi.QuotaFetchResponse, error) {
	if strings.TrimSpace(req.Provider) != ProviderID {
		return pluginapi.QuotaFetchResponse{}, fmt.Errorf("quota credential is not owned by ollama")
	}
	if p.manager == nil {
		return pluginapi.QuotaFetchResponse{}, fmt.Errorf("quota provider unavailable")
	}
	bridge := p.manager.bridge.withHostCallbackID(p.hostCallbackID)
	storageJSON := req.StorageJSON
	if len(storageJSON) == 0 {
		if strings.TrimSpace(req.AuthIndex) == "" || bridge == nil {
			return pluginapi.QuotaFetchResponse{}, fmt.Errorf("ollama quota credential storage is unavailable")
		}
		var err error
		storageJSON, err = bridge.AuthGet(ctx, req.AuthIndex)
		if err != nil {
			return pluginapi.QuotaFetchResponse{}, fmt.Errorf("ollama quota credential storage is unavailable")
		}
	}
	key, err := quotaCredential(req.Attributes, storageJSON)
	if err != nil {
		return pluginapi.QuotaFetchResponse{}, err
	}
	p.manager.mu.RLock()
	baseURL, timeout := p.manager.cfg.BaseURL, p.manager.cfg.RequestTimeout
	p.manager.mu.RUnlock()
	usage, err := fetchQuota(ctx, bridge, baseURL, timeout, key)
	if err != nil {
		return pluginapi.QuotaFetchResponse{}, err
	}
	return nativeQuotaResponse(usage), nil
}

func (quotaProvider) ResetQuota(context.Context, pluginapi.QuotaResetRequest) (pluginapi.QuotaResetResponse, error) {
	return pluginapi.QuotaResetResponse{}, fmt.Errorf("ollama quota reset is unsupported")
}

func quotaCredential(attributes map[string]string, storageJSON []byte) (string, error) {
	var stored struct {
		Type     string `json:"type"`
		Provider string `json:"provider"`
		APIKey   string `json:"api_key"`
	}
	if len(storageJSON) == 0 || json.Unmarshal(storageJSON, &stored) != nil {
		return "", fmt.Errorf("ollama quota credential storage is invalid")
	}
	typeID := strings.TrimSpace(stored.Type)
	providerID := strings.TrimSpace(stored.Provider)
	if (typeID != "" && typeID != ProviderID) || (providerID != "" && providerID != ProviderID) ||
		(typeID == "" && providerID == "") {
		return "", fmt.Errorf("quota credential is not owned by ollama")
	}
	key := strings.TrimSpace(stored.APIKey)
	if key == "" {
		return "", fmt.Errorf("ollama quota credential has no api key")
	}
	if attributeKey := strings.TrimSpace(attributes["api_key"]); attributeKey != "" && attributeKey != key {
		return "", fmt.Errorf("ollama quota credential is inconsistent")
	}
	return key, nil
}

// nativeQuotaResponse maps the management-center snapshot into the host ABI's
// quota shape: numeric balance/currency metrics, one bucket per metered
// window, and the 24h usage totals as a text metric.
func nativeQuotaResponse(usage quotaUsage) pluginapi.QuotaFetchResponse {
	metrics := make([]pluginapi.QuotaMetric, 0, len(usage.Metrics)+2)
	if usage.BalanceUSD > 0 {
		metrics = append(metrics, pluginapi.QuotaMetric{
			Key:      "balance_usd",
			Label:    "Balance",
			Value:    usage.BalanceUSD,
			Unit:     "USD",
			Format:   "currency",
			Currency: "USD",
		})
	}
	if usage.PurchasedUSD > 0 {
		metrics = append(metrics, pluginapi.QuotaMetric{
			Key:      "purchased_balance_usd",
			Label:    "Purchased balance",
			Value:    usage.PurchasedUSD,
			Unit:     "USD",
			Format:   "currency",
			Currency: "USD",
		})
	}
	for _, m := range usage.Metrics {
		metrics = append(metrics, textMetric(m.Label, m.Value))
	}
	buckets := make([]pluginapi.QuotaBucket, 0, len(usage.Rows))
	for _, row := range usage.Rows {
		buckets = append(buckets, nativeQuotaBucket(row))
	}
	resp := pluginapi.QuotaFetchResponse{Summary: metrics}
	if len(buckets) > 0 {
		resp.Groups = []pluginapi.QuotaGroup{{DisplayName: "Ollama quota", Buckets: buckets}}
	}
	return resp
}

// textMetric wraps one string-valued fact as an ABI metric (usage totals).
// The ABI's Format enum is "number"/"currency", so text facts ride in the
// label with a numeric zero value.
func textMetric(label, value string) pluginapi.QuotaMetric {
	return pluginapi.QuotaMetric{Key: metricKey(label), Label: label + ": " + value, Format: "number"}
}

// metricKey derives the stable ABI key from the display label.
func metricKey(label string) string {
	key := strings.ToLower(label)
	key = strings.ReplaceAll(key, " ", "_")
	key = strings.ReplaceAll(key, "(", "")
	key = strings.ReplaceAll(key, ")", "")
	return key
}

// nativeQuotaBucket converts a page row into the ABI bucket. Rows with a
// usable cap (USD allowance) carry a numeric remaining fraction; the legacy
// percent windows fall back to the same fraction via their percent.
func nativeQuotaBucket(row quotaRow) pluginapi.QuotaBucket {
	out := pluginapi.QuotaBucket{Window: row.Name, ResetTime: row.ResetsAt, Description: row.Label}
	if row.Text != "" {
		out.Description = row.Label + " (" + row.Text + ")"
	}
	switch {
	case row.Cap > 0:
		out.RemainingFraction = math.Max(0, math.Min(1, 1-row.Used/row.Cap))
	case row.Percent > 0:
		out.RemainingFraction = math.Max(0, math.Min(1, 1-float64(row.Percent)/100))
	default:
		out.RemainingFraction = 1
	}
	return out
}
