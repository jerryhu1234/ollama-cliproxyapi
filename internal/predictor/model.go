package predictor

import (
	"math"
	"time"
)

// RateSource is the estimation seam (plan D3): P1 ships only the sampling
// estimator; a telemetry-backed source may join later without touching the
// state machine or views. Units are the window's own units per hour
// (USD/h or percent/h).
type RateSource interface {
	// Estimate returns the conservative and optimistic burn rates plus
	// diagnostics for one window series. unit selects epsilon semantics
	// (UnitUSD / UnitPct). conservative is the faster slope (plan D6:
	// schedule against the worst recent window); optimistic is the
	// long-window slope (never faster than the short one, floor epsilon).
	Estimate(unit string, series []WindowSample, now time.Time) (cons, opt, r2 float64, samples int)
}

// SamplingRateSource estimates rates from balance-sampled series only.
type SamplingRateSource struct{}

// Reset-bump tolerance: a remaining RISE beyond this marks a window reset
// and starts a new segment.
const (
	bumpPctOfCap = 0.05
	bumpUSDFloor = 0.5
	bumpPct      = 5
)

// Window-unit epsilons: below these rates a window reads as "stable".
const (
	epsUSDRate     = 0.005 // $ per hour
	epsUSDZero     = 0.01  // $
	epsPctRate     = 0.05  // percent per hour
	epsPctZero     = 0.5   // percent
	etaWalkMaxHour = 768   // 32 days — beyond any Ollama reset period
)

// unitEps returns the (exhausted-zero, stable-rate) epsilons for a unit.
func unitEps(unit string) (epsZero, epsRate float64) {
	if unit == UnitPct {
		return epsPctZero, epsPctRate
	}
	return epsUSDZero, epsUSDRate
}

// currentSegment returns the samples of the CURRENT window segment:
// everything after the last reset bump. A bump is a remaining rise beyond
// tolerance, or a forwards jump of resets_at by more than half a day
// (period rolled over).
func currentSegment(unit string, series []WindowSample, now time.Time) []WindowSample {
	_ = now
	if len(series) < 2 {
		return series
	}
	for i := len(series) - 1; i > 0; i-- {
		prev, cur := series[i-1], series[i]
		if cur.Remaining > prev.Remaining+bumpTolerance(unit, cur.Cap) {
			return series[i:]
		}
		if !cur.ResetsAt.Equal(prev.ResetsAt) && cur.ResetsAt.Sub(prev.ResetsAt) > 12*time.Hour {
			return series[i:]
		}
	}
	return series
}

func bumpTolerance(unit string, cap float64) float64 {
	if unit == UnitPct {
		return bumpPct
	}
	tol := bumpPctOfCap * cap
	if tol < bumpUSDFloor {
		tol = bumpUSDFloor
	}
	return tol
}

// Estimate implements RateSource on sampled series with two trailing
// windows: short τ=6h (responsive, plan r_short) and long τ=24h or
// segment span (plan r_long, stabilizer). Plan D6 pairing: conservative is
// the faster of the two; optimistic is the long window itself.
func (SamplingRateSource) Estimate(unit string, series []WindowSample, now time.Time) (cons, opt, r2 float64, samples int) {
	if len(series) < 2 {
		return 0, 0, 0, len(series)
	}
	seg := currentSegment(unit, series, now)
	if len(seg) < 2 {
		return 0, 0, 1, len(seg)
	}
	short := trailingFit(seg, now, 6*time.Hour)
	long := trailingFit(seg, now, 24*time.Hour)
	shortBurn := math.Max(0, -short.b)
	longBurn := math.Max(0, -long.b)
	cons = math.Max(shortBurn, longBurn)
	opt = longBurn
	r2, samples = short.r2, len(seg)
	return cons, opt, r2, samples
}

type fit struct {
	b  float64 // hourly slope of remaining (negative while burning)
	r2 float64
}

// trailingFit is an OLS slope over the trailing window of a series. When
// fewer than two points fall inside the window (early life), it defers to
// the last two points of the segment so a conservative slope still exists.
func trailingFit(series []WindowSample, now time.Time, window time.Duration) fit {
	end := len(series) - 1
	cutoff := now.Add(-window)
	start := 0
	for i := range series {
		if series[i].Ts.After(cutoff) {
			start = i
			break
		}
	}
	if end-start < 1 {
		start = end - 1
	}
	if start < 0 {
		start = 0
	}
	return ols(series[start : end+1])
}

// ols fits remaining ~ hours-since-first-sample; x in hours keeps the
// slope unit "remaining per hour" directly.
func ols(series []WindowSample) fit {
	n := len(series)
	if n < 2 {
		return fit{}
	}
	var x0 = series[0].Ts
	var sx, sy, sxx, sxy float64
	for _, sm := range series {
		x := sm.Ts.Sub(x0).Hours()
		sx += x
		sy += sm.Remaining
		sxx += x * x
		sxy += x * sm.Remaining
	}
	nf := float64(n)
	den := nf*sxx - sx*sx
	if den <= 1e-12 {
		return fit{b: 0, r2: 1} // degenerate: all points at one instant
	}
	b := (nf*sxy - sx*sy) / den
	meanY := sy / nf
	var ssRes, ssTot float64
	for _, sm := range series {
		x := sm.Ts.Sub(x0).Hours()
		pred := (sy-b*sx)/nf + b*x
		ssRes += (sm.Remaining - pred) * (sm.Remaining - pred)
		ssTot += (sm.Remaining - meanY) * (sm.Remaining - meanY)
	}
	r2 := 1.0
	if ssTot > 1e-9 {
		r2 = math.Max(0, 1-ssRes/ssTot)
	}
	return fit{b: b, r2: r2}
}

// shapeScale returns the per-hour multiplier baseline of a profile: steps
// are computed as scale × p[hod] so their window-average equals the rate
// anchor. Returns 0 when the profile is degenerate (caller then falls
// back to flat rate).
func shapeScale(profile map[int]float64) float64 {
	if len(profile) == 0 {
		return 0
	}
	mass := 0.0
	for _, v := range profile {
		if v > 0 {
			mass += v
		}
	}
	if mass <= 0 {
		return 0
	}
	return 24.0 / mass
}

// walkEta projects one window forward hour by hour under the shape-aware
// model (plan §4): rate(t) = rate × 24 × p[hod(t)] / Σp — the window-
// average burn stays equal to the conservative slope while the hourly
// shape modulates WHEN the burn happens. profile==nil degrades to flat
// rate (pct windows and profile-less accounts). Returns elapsed until
// exhaustion (or the walk cap), the burn accumulated in that span, and
// whether the window exhausts within the walk.
func walkEta(remaining, rate float64, profile map[int]float64, from, resets time.Time, maxHours int) (eta time.Duration, burn float64, exhausts bool) {
	if remaining <= 0 {
		return 0, 0, true
	}
	scale := 1.0
	if profile != nil {
		s := shapeScale(profile)
		if s <= 0 {
			profile = nil
		} else {
			scale = s
		}
	}
	t := from
	elapsed := time.Duration(0)
	total := 0.0
	for i := 0; i < maxHours; i++ {
		h := 0.0
		if profile != nil {
			h = profile[t.UTC().Hour()]
			if h < 0 {
				h = 0
			}
		}
		step := rate * scale * h
		if profile == nil {
			step = rate
		}
		if step >= remaining && step > 0 {
			// Consumed mid-hour: fractional completion keeps ETAs honest.
			elapsed += time.Duration(float64(time.Hour) * remaining / step)
			return elapsed, total + remaining, true
		}
		remaining -= step
		total += step
		elapsed += time.Hour
		t = t.Add(time.Hour)
		if !resets.IsZero() && !t.Before(resets) {
			return elapsed, total, false
		}
	}
	return elapsed, total, remaining <= 0
}
