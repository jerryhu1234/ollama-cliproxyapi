package predictor

import (
	"math"
	"time"
)

// fleetWindow is the per-account projection state for one metered window.
type fleetWindow struct {
	unit      string
	remaining float64 // usd or pct
	cap       float64 // usd only
	resets    time.Time
	rate      float64 // conservative, per hour, in window units
	published bool    // window exists for this account
	samples   int     // observation count backing the rate
}

// aggregateInput is one active account's projection seeds.
type aggregateInput struct {
	keyID   string
	windows map[string]fleetWindow
}

// FleetOptions bound the fleet projection.
type FleetOptions struct {
	Now        time.Time
	Horizon    time.Duration
	StaleAfter time.Duration
	Src        RateSource
}

// collectInputs reads the store into per-account projection seeds for every
// non-stale key. Stale accounts are excluded from rates/ETA/projection but
// counted in the total (plan §5 stale rule).
func collectInputs(store *Store, keyIDs []string, opts FleetOptions) (active []aggregateInput, staleCount int) {
	for _, keyID := range keyIDs {
		if len(store.WindowNames(keyID)) == 0 {
			continue
		}
		if store.Stale(keyID, opts.StaleAfter) {
			staleCount++
			continue
		}
		in := aggregateInput{keyID: keyID, windows: make(map[string]fleetWindow)}
		for _, name := range store.WindowNames(keyID) {
			series := store.Series(keyID, name)
			if len(series) == 0 {
				continue
			}
			last := series[len(series)-1]
			unit := windowUnit(name)
			cons, _, _, _ := opts.Src.Estimate(unit, series, opts.Now)
			in.windows[name] = fleetWindow{
				unit:      unit,
				remaining: last.Remaining,
				cap:       last.Cap,
				resets:    effectiveResets(last.ResetsAt, opts.Now),
				rate:      cons,
				published: true,
				samples:   len(series),
			}
		}
		if len(in.windows) > 0 {
			active = append(active, in)
		}
	}
	return active, staleCount
}

// fleetUSDWindows returns the USD-bearing windows of an input summed.
func fleetUSD(in aggregateInput) (remaining, cap, rate float64) {
	for name, w := range in.windows {
		if w.unit != UnitUSD {
			continue
		}
		remaining += math.Max(0, w.remaining)
		rate += math.Max(0, w.rate)
		if name == WindowIncluded {
			cap += w.cap
		}
	}
	return remaining, cap, rate
}

// simWindow is one window's mutable state during the step walk.
type simWindow struct {
	name   string
	rem    float64
	rate   float64 // conservative, per hour, in window units
	cap    float64 // usd only
	unit   string
	resets time.Time
}

// accountStateAt reports the account's projected state at instant t (plan §5
// projection + §12 D7 gating): a 1h stepped walk under flat conservative
// rates with reset bumps.
//
// Hard-gate coupling (D7): the ollama Pro windows share one consumption
// stream — the same requests drain the 5h session AND the weekly meter — so
// once either pct meter reaches empty the account stops serving entirely and
// ALL windows freeze (a weekly exhaustion suspends the 5h window even at
// 46%) until the gate's reset bump refills it. Max accounts (single monthly
// USD window) carry no pct gate and project exactly as before. The included
// window stays the serving gate as shipped: included at empty marks the
// account inactive even when a purchased balance could still bill.
//
// Reset semantics stay as shipped: a window whose reset instant lies behind
// the step stamp is bumped back to cap/100% at every such step (post-reset
// windows hold full capacity within the horizon). epsRate is the estimator's
// stable-window epsilon; the walk burns only on strictly positive rates, so
// stable windows behave identically without it.
func accountStateAt(in aggregateInput, until, now time.Time, epsZero float64) (usdLeft float64, active bool) {
	wins := make([]simWindow, 0, len(in.windows))
	for name, w := range in.windows {
		wins = append(wins, simWindow{
			name:   name,
			rem:    w.remaining,
			rate:   math.Max(0, w.rate),
			cap:    w.cap,
			unit:   w.unit,
			resets: w.resets,
		})
	}
	if !until.After(now) {
		return accountSnapshotState(wins, epsZero)
	}
	const maxSteps = 24 * 31 // hard walk bound (~1 month)
	ts := now
	for step := 0; step < maxSteps && ts.Before(until); step++ {
		stamp := ts.Add(time.Hour)
		if stamp.After(until) {
			stamp = until
		}
		hh := stamp.Sub(ts).Hours()
		// Shared-consumption gate on current rems: ANY empty pct meter
		// freezes the account for this step (D7).
		if pctGatesOpen(wins, epsZero) {
			for i := range wins {
				w := &wins[i]
				if w.rem > 0 && w.rate > 0 {
					w.rem = math.Max(0, w.rem-w.rate*hh)
				}
			}
		}
		// Reset bump last: the refill wins the stamp (stay-full semantics
		// as shipped).
		for i := range wins {
			w := &wins[i]
			if !w.resets.IsZero() && stamp.After(w.resets) {
				if w.cap > 0 {
					w.rem = w.cap
				} else if w.unit == UnitPct {
					w.rem = 100
				}
			}
		}
		ts = stamp
	}
	return accountSnapshotState(wins, epsZero)
}

// pctGatesOpen reports whether every pct hard gate (session/weekly) holds
// capacity at the current simulated rems.
func pctGatesOpen(wins []simWindow, epsZero float64) bool {
	for _, w := range wins {
		if w.unit == UnitPct && w.rem <= epsZero {
			return false
		}
	}
	return true
}

// accountSnapshotState snapshots the walk state: usable USD total and the
// serving flag (any empty pct gate, or the included window at empty, stops
// the account).
func accountSnapshotState(wins []simWindow, epsZero float64) (usdLeft float64, active bool) {
	active = true
	for _, w := range wins {
		switch {
		case w.unit == UnitUSD:
			usdLeft += math.Max(0, w.rem)
			if w.name == WindowIncluded && w.rem <= epsZero {
				active = false
			}
		case w.unit == UnitPct && w.rem <= epsZero:
			active = false
		}
	}
	return usdLeft, active
}

// FleetView computes the type-level aggregate (plan §5) from the store.
func BuildFleetView(store *Store, keyIDs []string, opts FleetOptions, deadline *time.Time) FleetView {
	view := FleetView{TotalAccounts: len(keyIDs), Risk: RiskHealthy, Projection: []ProjectionPoint{}}
	acc, stale := collectInputs(store, keyIDs, opts)
	view.StaleAccounts = stale
	epsZero, epsRate := unitEps(UnitUSD)

	var sumRem, sumCap, sumRate, liveRem, liveRate float64
	minTime := time.Duration(math.MaxInt64)
	firstDepletion := time.Duration(0)
	deadlineView := DeadlineView{}

	steps := int(opts.Horizon / time.Hour)
	if steps < 1 {
		steps = 1
	}
	// hourly projection accumulators
	stepUSD := make([]float64, steps)
	stepActive := make([]int, steps)

	for _, in := range acc {
		rem, cap, rate := fleetUSD(in)
		sumRem += rem
		sumCap += cap
		sumRate += rate
		// Dark derivation (plan §12 D-status) drives the fleet card's
		// waiting-reset tally and keeps the ETA denominators honest: a dark
		// account holds no capacity, and its leftover burn estimates would
		// skew Σrem/Σrate toward immediate depletion.
		var gates []darkGate
		for name, w := range in.windows {
			gates = append(gates, darkGate{name: name, unit: w.unit, remaining: w.remaining, resets: w.resets, samples: w.samples})
		}
		if dark, _ := evaluateDark(gates); dark {
			view.ExhaustedAccounts++
		} else {
			liveRem += rem
			liveRate += rate
		}
		// First depletion: earliest exhausts-in window of the account.
		for name, w := range in.windows {
			_ = name
			if !w.published || w.remaining <= epsZero {
				continue
			}
			if w.rate <= epsRate {
				continue // stable window never depletes
			}
			eta, _, exhausts := walkEta(w.remaining, w.rate, nil, opts.Now, w.resets, etaWalkMaxHour)
			if exhausts && eta > 0 && eta < minTime {
				minTime = eta
			}
		}
		// Projection walk (flat conservative rates with reset bumps, plan §5).
		for s := 0; s < steps; s++ {
			t := opts.Now.Add(time.Duration(s+1) * time.Hour)
			usdLeft, active := accountStateAt(in, t, opts.Now, epsZero)
			stepUSD[s] += usdLeft
			if active {
				stepActive[s]++
			}
		}
	}
	view.ActiveAccounts = len(acc)

	for s := 0; s < steps; s++ {
		ts := opts.Now.Add(time.Duration(s+1) * time.Hour)
		view.Projection = append(view.Projection, ProjectionPoint{
			float64(ts.UnixMilli()), round2(stepUSD[s]), float64(stepActive[s]),
		})
	}
	// Deadline evaluation (plan §5): one exact per-account projection at
	// the requested instant, independent of the hourly grid, so the card
	// answers "how much capacity at T" rather than a sum over steps.
	if deadline != nil && !deadline.IsZero() && deadline.After(opts.Now) {
		for _, in := range acc {
			usdLeft, active := accountStateAt(in, *deadline, opts.Now, epsZero)
			deadlineView.AvailableUSD += usdLeft
			if active {
				deadlineView.AvailableAccounts++
			}
		}
		deadlineView.Deadline = FormatTimestamp(*deadline)
		if sumCap > 0 {
			deadlineView.Fraction = round2(deadlineView.AvailableUSD / sumCap)
		}
		view.AtDeadline = &deadlineView
	}
	if liveRem > 0 && liveRate > 0 {
		view.FleetETA = etaSeconds(time.Duration(liveRem / liveRate * float64(time.Hour)))
	}
	if minTime < math.MaxInt64 {
		firstDepletion = minTime
		view.FirstDepletion = etaSeconds(firstDepletion)
	}
	view.RemainingUSD = round2(sumRem)
	view.CapUSD = round2(sumCap)
	view.RatePerHour = round2(sumRate)
	view.Risk = fleetRisk(view.ExhaustedAccounts, view.ActiveAccounts, sumRem, sumCap, view.Projection, epsZero)
	return view
}

// Dark-fraction thresholds for the fleet badge (D9): waiting-for-reset is
// routine in a pool with rolling weekly windows, so a single dark account
// must not cry wolf — half the fleet dark is a crisis, a fifth is pressure.
const (
	riskDarkCritical = 0.5
	riskDarkTight    = 0.2
)

// fleetRisk resolves the fleet badge (plan §5 + §12 D9): two axes, worst
// wins. Capacity: <10% remaining is critical; horizon-end <30% tight.
// Dark: the D8 waiting-reset share of sampled accounts — ≥50% critical,
// ≥20% tight. The shipped any-account-drained and first-depletion-inside-
// 24h triggers are gone: Pro-shaped accounts (no included window, purchased
// at $0) sat on "zero" forever, and in a rolling-reset pool some account
// always dies within 24h — both fired permanently on normal operation.
func fleetRisk(exhausted, active int, sumRem, sumCap float64, projection []ProjectionPoint, epsZero float64) string {
	fraction := 1.0
	if sumCap > 0 {
		fraction = sumRem / sumCap
	}
	darkFrac := 0.0
	if active > 0 {
		darkFrac = float64(exhausted) / float64(active)
	}
	if fraction < 0.10 || darkFrac >= riskDarkCritical {
		return RiskCritical
	}
	if len(projection) > 0 && sumCap > 0 {
		endUSD := 0.0
		for _, p := range projection {
			endUSD = p[1] // last point wins
		}
		if endUSD/sumCap < 0.30 {
			return RiskTight
		}
	}
	if darkFrac >= riskDarkTight {
		return RiskTight
	}
	return RiskHealthy
}
