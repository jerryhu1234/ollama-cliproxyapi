// Package predictor implements the usage prediction engine described in
// docs/usage-prediction-plan.md: a throttled balance/usage sampler feeding a
// local time-series store, from which per-account window ETAs and the
// provider-type fleet projection are computed. Page/API reads NEVER touch
// the upstream account API — sampling is the only upstream traffic path,
// bounded by a global token budget.
//
// The package is deliberately provider-agnostic: the plugin layer binds
// FetchFn (account-API → AccountSnapshot) and stores digest identities as
// KeyID; raw credentials never enter this package beyond the FetchFn call
// argument, and never reach persistence or logs.
package predictor

import (
	"context"
	"time"
)

// Metered window names as reported by the account API shapes.
const (
	WindowIncluded  = "included"  // USD allowance: cap + remaining + period end
	WindowSession   = "session"   // legacy percent window (rolling)
	WindowWeekly    = "weekly"    // legacy percent window
	WindowPurchased = "purchased" // purchased USD balance: no cap, no reset
)

// Window units for the API surface.
const (
	UnitUSD = "usd"
	UnitPct = "pct"
)

// Window states (plan §4).
const (
	StateExhaustedIn = "exhausts-in"  // projected depletion before the next reset
	StateResetsFirst = "resets-first" // window resets before projected depletion
	StateExhausted   = "exhausted"    // already at/under zero, waiting for reset
	StateStable      = "stable"       // consumed before, currently zero burn
	StateNoData      = "no-data"      // fewer than three samples in segment
	StateNone        = "none"         // F2: capless billed balance (purchased) empty — money has no reset cycle
)

// Confidence levels (plan §6); display precision degrades with the level.
const (
	ConfidenceLow    = "low"
	ConfidenceMedium = "medium"
	ConfidenceHigh   = "high"
)

// Risk badges for the fleet view (plan §5).
const (
	RiskHealthy  = "healthy"
	RiskTight    = "tight"
	RiskCritical = "critical"
)

// Account status markers (plan §12 D-status): the derived now-state of an
// account from its gate windows — ready, or dark waiting for a reset.
const (
	StatusReady     = "ready"
	StatusExhausted = "exhausted"
)

// USDWindow is one USD-metered window: the included allowance with its
// monthly reset, when the account shape carries it.
type USDWindow struct {
	Remaining float64
	Cap       float64
	ResetsAt  time.Time // zero when the upstream omits the period
}

// PctWindow is one legacy remaining-percentage window.
type PctWindow struct {
	RemainingPercent float64
	ResetsAt         time.Time
}

// HourUsage is one upstream usage bucket. The 24h range yields hourly
// buckets; the trailing bucket is Partial and systematically UNDER-counts
// upstream ("new usage may take time to appear"), so profiles never weight
// it. Buckets provide the diurnal burn SHAPE; the balance delta remains the
// authoritative rate anchor (plan §4 shape-aware model).
type HourUsage struct {
	HourStart time.Time // hour-aligned UTC
	UsageUSD  float64
	Partial   bool
}

// AccountSnapshot is one account observation handed to the engine by the
// plugin-side FetchFn. Absent windows are nil / negative.
type AccountSnapshot struct {
	KeyID        string    // digest identity; never the raw credential
	Label        string    // human label (digest-derived); persisted as-is
	At           time.Time // observation time (UTC)
	Included     *USDWindow
	Session      *PctWindow // legacy; optional
	Weekly       *PctWindow // legacy; optional
	PurchasedUSD float64    // present when >= 0; -1 when absent upstream
	Hours        []HourUsage
}

// FetchFn performs one upstream account observation for one credential.
// Implementations return the snapshot mapped from /api/balance + /api/usage.
type FetchFn func(ctx context.Context, key string) (AccountSnapshot, error)

// KeyRef binds a credential to its store identity. Key is passed ONLY to
// FetchFn; it must never be logged or persisted.
type KeyRef struct {
	ID    string
	Label string
	Key   string
}

// WindowSample is one observation of one window series.
type WindowSample struct {
	Ts        time.Time
	Remaining float64 // USD (usd/purchased windows) or percent (pct windows)
	Cap       float64 // USD cap (usd window); zero otherwise
	ResetsAt  time.Time
}

// WindowID addresses one series in the store.
type WindowID struct {
	KeyID string
	Name  string
}
