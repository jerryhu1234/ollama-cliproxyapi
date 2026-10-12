package predictor

import (
	"math"
	"time"
)

// WindowView is the API face of one metered window (plan §6 response
// shape): remaining/cap, conservative + optimistic rates and ETAs, the
// resolved state and confidence, and the bounded sparkline series.
type WindowView struct {
	Name          string       `json:"name"`
	Unit          string       `json:"unit"`
	Remaining     float64      `json:"remaining"`
	Cap           float64      `json:"cap,omitempty"`
	ResetsAt      string       `json:"resets_at,omitempty"`
	State         string       `json:"state"`
	Confidence    string       `json:"confidence"`
	RatePerHour   float64      `json:"rate_per_hour"`
	OptimisticPH  float64      `json:"optimistic_rate_per_hour,omitempty"`
	Eta           float64      `json:"eta_seconds"`
	OptimisticETA float64      `json:"optimistic_eta_seconds,omitempty"`
	BurnToReset   float64      `json:"burn_to_reset,omitempty"`
	Samples       int          `json:"samples"`
	R2            float64      `json:"r2,omitempty"`
	ShapeAware    bool         `json:"shape_aware,omitempty"`
	Accelerated   bool         `json:"accelerated,omitempty"`
	GatedBy       string       `json:"gated_by,omitempty"`
	GatedAt       string       `json:"gated_at,omitempty"`
	Series        [][2]float64 `json:"series,omitempty"`
}

// AccountView is one credential's prediction card.
type AccountView struct {
	KeyID     string       `json:"key_id"`
	Label     string       `json:"label"`
	UpdatedAt string       `json:"updated_at"`
	Stale     bool         `json:"stale"`
	Status    string       `json:"status,omitempty"`
	NextReset string       `json:"next_reset,omitempty"`
	Windows   []WindowView `json:"windows"`
}

// FleetView is the provider-type aggregate (plan §5): totals, fleet ETA,
// first depletion, risk badge, the hourly capacity projection and the
// optional deadline snapshot.
type FleetView struct {
	RemainingUSD      float64           `json:"remaining_usd"`
	CapUSD            float64           `json:"cap_usd"`
	RatePerHour       float64           `json:"rate_per_hour"`
	FleetETA          float64           `json:"fleet_eta_seconds"`
	FirstDepletion    float64           `json:"first_depletion_seconds"`
	ActiveAccounts    int               `json:"active_accounts"`
	StaleAccounts     int               `json:"stale_accounts"`
	TotalAccounts     int               `json:"total_accounts"`
	ExhaustedAccounts int               `json:"exhausted_accounts,omitempty"`
	Risk              string            `json:"risk"`
	Projection        []ProjectionPoint `json:"projection"`
	AtDeadline        *DeadlineView     `json:"at_deadline,omitempty"`
}

// ProjectionPoint is [unix_ms, available_usd, active_accounts].
type ProjectionPoint [3]float64

// DeadlineView answers "how much capacity remains at timestamp T" (plan §5).
type DeadlineView struct {
	Deadline          string  `json:"deadline"`
	AvailableAccounts int     `json:"available_accounts"`
	AvailableUSD      float64 `json:"available_usd"`
	Fraction          float64 `json:"fraction"`
}

// windowUnit resolves the unit semantics by window name (series alone
// cannot distinguish tiny-USD from percent).
func windowUnit(name string) string {
	switch name {
	case WindowSession, WindowWeekly:
		return UnitPct
	default:
		return UnitUSD
	}
}

// seriesTail keeps the sparkline bounded to the most recent points.
const seriesTail = 48

// buildWindowView computes one window's prediction view. profile carries
// the account's hourly-burn shape when the unit is USD and coverage exists
// (shape-aware model, plan §4); pct windows always run flat-rate.
func buildWindowView(name string, series []WindowSample, now time.Time, profile map[int]float64, src RateSource, accelerated bool) WindowView {
	unit := windowUnit(name)
	view := WindowView{Name: name, Unit: unit, Samples: len(series)}
	if len(series) == 0 {
		view.State, view.Confidence = StateNoData, ConfidenceLow
		return view
	}
	last := series[len(series)-1]
	view.Remaining = last.Remaining
	if unit == UnitUSD {
		view.Cap = last.Cap
	}
	resets := effectiveResets(last.ResetsAt, now)
	view.ResetsAt = FormatTimestamp(resets)
	view.Series = tailSeries(series)
	view.Accelerated = accelerated

	cons, opt, r2, samples := src.Estimate(unit, series, now)
	epsZero, epsRate := unitEps(unit)
	stable := cons <= epsRate && math.Max(0, last.Remaining) > epsZero

	switch {
	case last.Remaining <= epsZero:
		// F2: the capless USD window is the purchased billing channel —
		// money has no reset cycle, so an empty balance reads neutral
		// instead of "exhausted · waiting for reset".
		if name == WindowPurchased {
			view.State = StateNone
			view.Confidence = classifyConfidence(samples, r2, true)
			view.RatePerHour, view.OptimisticPH = cons, opt
			view.Samples, view.R2 = samples, round2(r2)
			return view
		}
		view.State = StateExhausted
		view.Confidence = classifyConfidence(samples, r2, true)
		view.RatePerHour, view.OptimisticPH = cons, opt
		view.Samples, view.R2 = samples, round2(r2)
		return view
	case samples < 3:
		view.State = StateNoData
		view.Confidence = ConfidenceLow
		view.RatePerHour, view.OptimisticPH = cons, opt
		view.Samples, view.R2 = samples, round2(r2)
		return view
	case stable:
		view.State = StateStable
		view.Confidence = classifyConfidence(samples, r2, true)
		view.RatePerHour, view.OptimisticPH = cons, opt
		view.Samples, view.R2 = samples, round2(r2)
		return view
	}

	prof := profile
	if unit != UnitUSD {
		prof = nil // legacy percent windows have no USD buckets upstream
	}
	// Exhaustion walk ignores the reset bump: eta is the raw drain time.
	eta, _, _ := walkEta(last.Remaining, cons, prof, now, time.Time{}, etaWalkMaxHour)
	etaOpt, _, _ := walkEta(last.Remaining, opt, prof, now, time.Time{}, etaWalkMaxHour)
	// Reset-bounded walk answers "burn before the next refill".
	if !resets.IsZero() {
		_, burn, exhaustsBefore := walkEta(last.Remaining, cons, prof, now, resets, etaWalkMaxHour)
		if !exhaustsBefore {
			view.State = StateResetsFirst
			view.BurnToReset = round2(burn)
		}
	}
	if view.State == "" {
		view.State = StateExhaustedIn
	}
	view.Confidence = classifyConfidence(samples, r2, false)
	view.RatePerHour = round2(cons)
	view.OptimisticPH = round2(opt)
	view.Eta = etaSeconds(eta)
	view.OptimisticETA = etaSeconds(etaOpt)
	view.Samples, view.R2 = samples, round2(r2)
	view.ShapeAware = prof != nil
	return view
}

// effectiveResets zeroes resets in the past (upstream lag) so they cannot
// poison the resets-first comparison.
func effectiveResets(resets, now time.Time) time.Time {
	if resets.IsZero() || resets.Before(now) {
		return time.Time{}
	}
	return resets
}

// tailSeries bounds the sparkline to the most recent seriesTail points,
// each [unix_ms, remaining].
func tailSeries(series []WindowSample) [][2]float64 {
	start := 0
	if len(series) > seriesTail {
		start = len(series) - seriesTail
	}
	out := make([][2]float64, 0, len(series)-start)
	for _, sm := range series[start:] {
		out = append(out, [2]float64{
			float64(sm.Ts.UnixMilli()), math.Max(0, sm.Remaining),
		})
	}
	return out
}

func round2(v float64) float64 {
	return math.Round(v*100) / 100
}

// darkGate is one window's gate observation for the account now-state: only
// windows with enough samples can gate (the same threshold the state
// machine applies); the unit epsilon decides the empty comparison.
type darkGate struct {
	name      string
	unit      string
	remaining float64
	resets    time.Time
	samples   int
}

// evaluateDark derives the account's dark-now state (plan §12 D-status):
// any pct hard gate (session/weekly — D7) at empty marks the account
// exhausted, and so does the included serving window — unless the purchased
// billing channel holds balance (F3/D8: the combined credits pool keeps
// burning paid balance, and the purchased side never refills, so its
// remaining buffer lifts included's dark mark; the pct gates are rolling
// rate meters on their own clocks, so the buffer never lifts them).
// Returns the flag and the earliest reset instant among the dark gates
// (zero when no gate carries one).
func evaluateDark(gates []darkGate) (dark bool, nextReset time.Time) {
	purchased := -1.0 // absent reading → no buffer: stay conservative
	for _, g := range gates {
		if g.name == WindowPurchased {
			purchased = g.remaining
		}
	}
	epsUSD, _ := unitEps(UnitUSD)
	for _, g := range gates {
		if g.samples < 3 {
			continue // below the state machine's no-data threshold
		}
		epsZero, _ := unitEps(g.unit)
		if g.remaining > epsZero {
			continue
		}
		if g.unit != UnitPct && g.name != WindowIncluded {
			continue // billing channels drain to zero freely
		}
		if g.name == WindowIncluded && purchased > epsUSD {
			continue // burning purchased credits, the account still serves
		}
		dark = true
		if !g.resets.IsZero() && (nextReset.IsZero() || g.resets.Before(nextReset)) {
			nextReset = g.resets
		}
	}
	return dark, nextReset
}

// buildAccountView assembles one account's window views from the store.
func buildAccountView(store *Store, keyID string, now time.Time, staleAfter time.Duration, src RateSource, accelerated map[string]bool) AccountView {
	view := AccountView{KeyID: keyID, Label: store.Label(keyID)}
	if view.Label == "" {
		view.Label = "Ollama credential " + shortDigest(keyID)
	}
	if t, ok := store.LastOK(keyID); ok {
		view.UpdatedAt = FormatTimestamp(t)
	}
	view.Stale = store.Stale(keyID, staleAfter)
	prof := profileFor(store, keyID)
	var gates []darkGate
	for _, name := range store.WindowNames(keyID) {
		series := store.Series(keyID, name)
		if len(series) == 0 {
			continue
		}
		last := series[len(series)-1]
		gates = append(gates, darkGate{
			name:      name,
			unit:      windowUnit(name),
			remaining: last.Remaining,
			resets:    effectiveResets(last.ResetsAt, now),
			samples:   len(series),
		})
		view.Windows = append(view.Windows, buildWindowView(
			name, series, now, prof, src, accelerated[keyID],
		))
	}
	applyGating(view.Windows, now)
	if len(gates) > 0 {
		if dark, nextReset := evaluateDark(gates); dark {
			view.Status = StatusExhausted
			view.NextReset = FormatTimestamp(nextReset)
		} else {
			view.Status = StatusReady
		}
	}
	return view
}

// hardGateWindow reports whether a window suspends shared-consumption usage
// when it empties (plan §12 D7): the session and weekly pct meters drain
// from the same request stream, so either hitting empty blocks the
// account's other windows until the gate resets.
func hardGateWindow(name string) bool {
	return name == WindowSession || name == WindowWeekly
}

// applyGating annotates the D7 coupling on account windows: when a hard
// gate empties BEFORE a sibling window's own depletion and stays empty
// past that depletion instant (no earlier refill), the sibling's projected
// drain is moot this cycle — its eta is replaced by the freeze instant.
func applyGating(windows []WindowView, now time.Time) {
	gateEmpty := func(g WindowView) (time.Time, bool) {
		switch g.State {
		case StateExhausted:
			return now, true // already frozen
		case StateExhaustedIn:
			if g.Eta > 0 {
				return now.Add(time.Duration(g.Eta) * time.Second), true
			}
		}
		return time.Time{}, false
	}
	for i := range windows {
		v := &windows[i]
		if v.State != StateExhaustedIn || v.Eta <= 0 {
			continue
		}
		vEnd := now.Add(time.Duration(v.Eta) * time.Second)
		gateName := ""
		var earliest time.Time
		for j := range windows {
			g := windows[j]
			if i == j || !hardGateWindow(g.Name) {
				continue
			}
			gEmpty, ok := gateEmpty(g)
			if !ok || !gEmpty.Before(vEnd) {
				continue
			}
			// A refill before the window would deplete only delays the
			// drain; keep the standalone eta in that case.
			if g.ResetsAt != "" {
				if rt, err := time.Parse(time.RFC3339, g.ResetsAt); err == nil && rt.Before(vEnd) {
					continue
				}
			}
			if gateName == "" || gEmpty.Before(earliest) {
				gateName, earliest = g.Name, gEmpty
			}
		}
		if gateName != "" {
			v.GatedBy = gateName
			v.GatedAt = FormatTimestamp(earliest)
		}
	}
}

// shortDigest trims a digest for label fallbacks.
func shortDigest(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

// profileFor binds the store's merged hourly coverage into a model profile.
func profileFor(store *Store, keyID string) map[int]float64 {
	profile, ok := store.HourProfile(keyID)
	if !ok || len(profile) == 0 {
		return nil
	}
	var mass float64
	for _, v := range profile {
		if v > 0 {
			mass += v
		}
	}
	if mass <= 0 {
		return nil
	}
	return profile
}

// classifyConfidence follows plan §6: ample strong-fit samples → high;
// viable fit → medium; else low. Stable windows with ample samples read
// high too — "won't exhaust" is then the confident finding.
func classifyConfidence(samples int, r2 float64, stable bool) string {
	switch {
	case samples >= 8 && (r2 >= 0.8 || stable):
		return ConfidenceHigh
	case samples >= 3 && (r2 >= 0.5 || stable):
		return ConfidenceMedium
	default:
		return ConfidenceLow
	}
}

// etaSeconds converts an ETA to display seconds; zero means unknown.
func etaSeconds(d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return d.Seconds()
}
