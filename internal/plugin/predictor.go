// Plugin-side predictor wiring (plan §8): engine construction from config,
// lifecycle start/stop, and the management POST handler. The engine's
// goroutines all join before shutdown proceeds (they hold host-callback
// in-flight counts), and page/API reads never touch the upstream.
package plugin

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"

	"ollama-cliproxyapi/internal/config"
	"ollama-cliproxyapi/internal/predictor"
)

// predictorStopTimeout bounds the engine join on shutdown/reconfigure. The
// engine's fetch calls ride host-callback in-flight accounting, so this
// must elapse BEFORE bridge.WaitForInFlight in handleShutdown. Package var
// so tests can shrink it.
var predictorStopTimeout = 10 * time.Second

// predictorKeys converts config credentials into engine refs (digest
// identity — the raw key travels only inside Fetch calls).
func predictorKeys(cfg config.Config) []predictor.KeyRef {
	refs := make([]predictor.KeyRef, 0, len(cfg.APIKeys))
	for _, k := range cfg.APIKeys {
		id, label := quotaIdentity(k.Value)
		refs = append(refs, predictor.KeyRef{ID: id, Label: label, Key: k.Value})
	}
	return refs
}

// predictorDataDir resolves the persistence root: "off" disables
// persistence, "" auto-resolves under the user home, explicit path passes
// through. Auto-resolution failure degrades to memory-only with a warn
// (plan D1: persistence is default-on but never blocks the plugin over a
// filesystem matter).
func predictorDataDir(cfg config.Config) string {
	switch {
	case cfg.Predictor.DataDir == config.PredictorDataDirOff:
		return ""
	case cfg.Predictor.DataDir != "":
		return cfg.Predictor.DataDir
	}
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return ""
	}
	return filepath.Join(home, config.DefaultPredictorDataDirSuffix)
}

// buildPredictor constructs the engine for one lifecycle; nil when the
// predictor is disabled or no credentials are configured.
func (m *Manager) buildPredictor(cfg config.Config) *predictor.Engine {
	if !cfg.Predictor.Enabled || len(cfg.APIKeys) == 0 {
		return nil
	}
	dataDir := predictorDataDir(cfg)
	if dataDir != "" {
		// Best-effort early MkdirAll so a bad path is visible in the log
		// at startup instead of at the first flush.
		if err := os.MkdirAll(dataDir, 0o755); err != nil {
			m.predictorLog("warn", "predictor: data dir unusable, persistence disabled",
				map[string]any{"dir": dataDir, "error": err.Error()})
			dataDir = ""
		}
	}
	return predictor.NewEngine(predictor.Options{
		Interval:        cfg.Predictor.SampleInterval,
		BudgetPerMinute: cfg.Predictor.BudgetPerMinute,
		History:         cfg.Predictor.History,
		Horizon:         cfg.Predictor.Horizon,
		DataDir:         dataDir,
		Keys:            predictorKeys(cfg),
		Fetch:           m.fetchPredictorSnapshot,
		Log:             m.predictorLog,
	})
}

// predictorLog adapts the host bridge log. Messages and fields are already
// digest-only by predictor convention (plan §10 data-safety boundary).
func (m *Manager) predictorLog(level, msg string, fields map[string]any) {
	if m.bridge == nil {
		return
	}
	_ = m.bridge.Log(level, msg, fields)
}

// swapPredictor installs a new engine (nil allowed) and returns the retired one.
func (m *Manager) swapPredictor(next *predictor.Engine) *predictor.Engine {
	m.mu.Lock()
	defer m.mu.Unlock()
	old := m.pred
	m.pred = next
	return old
}

// currentPredictor reads the active engine (nil when disabled or stopped).
func (m *Manager) currentPredictor() *predictor.Engine {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.pred
}

// stopPredictor retires the active engine with a bounded join and a final
// flush (the engine flushes once its loops have all exited).
func (m *Manager) stopPredictor() {
	if old := m.swapPredictor(nil); old != nil {
		old.Stop()
		if !old.WaitStopped(predictorStopTimeout) {
			m.predictorLog("warn", "predictor: engine did not stop within the budget", nil)
		}
	}
}

// predictorRequest is the body of POST .../prediction (plan §6):
//   - {} → read-only snapshot (never an upstream call);
//   - {"deadline":"RFC3339"} → adds the deadline evaluation card;
//   - {"refresh":true,"key_id":"..."} → schedules one in-budget re-sample.
type predictorRequest struct {
	Deadline string `json:"deadline,omitempty"`
	Refresh  bool   `json:"refresh,omitempty"`
	KeyID    string `json:"key_id,omitempty"`
}

// handlePredictorManagement serves POST /v0/management/plugins/<id>/prediction.
func (m *Manager) handlePredictorManagement(req pluginapi.ManagementRequest) (pluginapi.ManagementResponse, error) {
	var body predictorRequest
	if len(req.Body) > 0 {
		if err := json.Unmarshal(req.Body, &body); err != nil {
			return pluginapi.ManagementResponse{StatusCode: http.StatusBadRequest, Body: []byte(`{"error":"invalid request"}`)}, nil
		}
	}
	pred := m.currentPredictor()
	if pred == nil {
		return quotaJSON(emptyPredictorResponse(time.Now().UTC()))
	}
	if body.Refresh {
		if !pred.RequestRefresh(body.KeyID) {
			return pluginapi.ManagementResponse{StatusCode: http.StatusNotFound, Body: []byte(`{"error":"unknown key"}`)}, nil
		}
	}
	if body.Deadline != "" {
		dl := parseRFC3339Time(body.Deadline)
		if dl.IsZero() || !dl.After(time.Now()) {
			return pluginapi.ManagementResponse{StatusCode: http.StatusBadRequest, Body: []byte(`{"error":"deadline must be a future RFC3339 timestamp"}`)}, nil
		}
		return quotaJSON(pred.Snapshot(&dl))
	}
	return quotaJSON(pred.Snapshot(nil))
}

// emptyPredictorResponse is the disabled-engine shape: no accounts, the
// ollama fleet present-but-zeroed so the page renders its empty state.
func emptyPredictorResponse(now time.Time) predictor.Response {
	return predictor.Response{
		GeneratedAt: predictor.FormatTimestamp(now),
		Persistence: &predictor.PersistStatus{State: predictor.PersistOff, Detail: "predictor disabled"},
		Accounts:    []predictor.AccountView{},
		ByType:      map[string]predictor.FleetView{"ollama": {Projection: []predictor.ProjectionPoint{}}},
	}
}
