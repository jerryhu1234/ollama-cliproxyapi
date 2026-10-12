package predictor

import (
	"testing"
	"time"
)

// seedLinear seeds two observations for one account so the estimator sees
// (remPast → remNow) over hoursAgo hours; returns the implied rate.
func seedLinear(s *Store, id string, base time.Time, hoursAgo int, remPast, remNow, capUSD float64) {
	s.Apply(AccountSnapshot{KeyID: id, Label: "cred " + id, At: base.Add(-time.Duration(hoursAgo) * time.Hour),
		Included: &USDWindow{Remaining: remPast, Cap: capUSD}})
	s.Apply(AccountSnapshot{KeyID: id, Label: "cred " + id, At: base,
		Included: &USDWindow{Remaining: remNow, Cap: capUSD}})
}

func TestFleetAggregation(t *testing.T) {
	now := time.Now().UTC()
	s := NewStore(24 * time.Hour)
	seedLinear(s, "a", now, 4, 55, 50, 100) // 1.25$/h
	seedLinear(s, "b", now, 4, 34, 30, 100) // 1.00$/h

	view := BuildFleetView(s, []string{"a", "b"}, FleetOptions{
		Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{},
	}, nil)

	if view.TotalAccounts != 2 || view.ActiveAccounts != 2 || view.StaleAccounts != 0 {
		t.Fatalf("account tallies = %+v", view)
	}
	if absDiff(view.RemainingUSD, 80) > 0.01 || absDiff(view.CapUSD, 200) > 0.01 {
		t.Fatalf("remaining/cap = %v/%v", view.RemainingUSD, view.CapUSD)
	}
	if absDiff(view.RatePerHour, 2.25) > 0.02 {
		t.Fatalf("rate = %v", view.RatePerHour)
	}
	// Σrem/Σrate = 80/2.25 ≈ 35.6h.
	if absDiff(view.FleetETA, (80/2.25)*3600) > 600 {
		t.Fatalf("fleet eta = %v", view.FleetETA)
	}
	// First depletion: b exhausts in 30h (a would take 40h).
	if absDiff(view.FirstDepletion, 30*3600) > 600 {
		t.Fatalf("first depletion = %v", view.FirstDepletion)
	}
	// End of horizon: both drained → <30% → tight.
	if view.Risk != RiskTight {
		t.Fatalf("risk = %q, want tight", view.Risk)
	}
	if len(view.Projection) != 48 {
		t.Fatalf("projection points = %d", len(view.Projection))
	}
	// Deadline snapshot at now+10h: 37.5 + 20 = 57.5 USD, both live.
	dl := now.Add(10 * time.Hour)
	view = BuildFleetView(s, []string{"a", "b"}, FleetOptions{
		Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{},
	}, &dl)
	if view.AtDeadline == nil {
		t.Fatal("at_deadline missing")
	}
	if absDiff(view.AtDeadline.AvailableUSD, 57.5) > 0.02 || view.AtDeadline.AvailableAccounts != 2 {
		t.Fatalf("at_deadline = %+v", view.AtDeadline)
	}
}

func TestFleetCriticalWhenAccountDrained(t *testing.T) {
	// D9 badge: a drained account must be DARK under the D8 derivation —
	// 3 samples to clear the no-data threshold — and 1-of-2 dark (50%)
	// crosses the crisis threshold.
	now := time.Now().UTC()
	s := NewStore(24 * time.Hour)
	for i, rem := range []float64{10, 5, 0} {
		s.Apply(AccountSnapshot{KeyID: "a", At: now.Add(time.Duration(i-2) * time.Hour),
			Included: &USDWindow{Remaining: rem, Cap: 20}})
	}
	seedLinear(s, "b", now, 4, 34, 30, 100)
	view := BuildFleetView(s, []string{"a", "b"}, FleetOptions{
		Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{},
	}, nil)
	if view.Risk != RiskCritical {
		t.Fatalf("risk = %q, want critical (1/2 dark ≥ 50%%)", view.Risk)
	}
	if view.ExhaustedAccounts != 1 {
		t.Fatalf("exhausted accounts = %d, want 1", view.ExhaustedAccounts)
	}
}

func TestFleetResetBumpKeepsHealthy(t *testing.T) {
	now := time.Now().UTC()
	s := NewStore(24 * time.Hour)
	reset := now.Add(30 * time.Hour)
	s.Apply(AccountSnapshot{KeyID: "a", At: now.Add(-4 * time.Hour),
		Included: &USDWindow{Remaining: 55, Cap: 100, ResetsAt: reset}})
	s.Apply(AccountSnapshot{KeyID: "a", At: now,
		Included: &USDWindow{Remaining: 50, Cap: 100, ResetsAt: reset}})
	seedLinear(s, "b", now, 4, 34, 30, 100) // exhausts at 30h
	view := BuildFleetView(s, []string{"a", "b"}, FleetOptions{
		Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{},
	}, nil)
	// a refills to cap at +30h → horizon end holds 100 USD → 50% capacity.
	last := view.Projection[len(view.Projection)-1]
	if absDiff(last[1], 100) > 0.02 {
		t.Fatalf("projection end = %v, want 100 (reset bump)", last[1])
	}
	if last[2] != 1 {
		t.Fatalf("active accounts at end = %v, want 1 (b drained, a refilled)", last[2])
	}
	// 30h > 24h and end fraction 0.5 → healthy.
	if view.Risk != RiskHealthy {
		t.Fatalf("risk = %q, want healthy", view.Risk)
	}
	// a's exhaustion is masked by the reset: first depletion must come from b.
	if absDiff(view.FirstDepletion, 30*3600) > 600 {
		t.Fatalf("first depletion = %v, want b's 30h", view.FirstDepletion)
	}
}

func TestFleetStaleExcluded(t *testing.T) {
	now := time.Now().UTC()
	s := NewStore(24 * time.Hour)
	seedLinear(s, "a", now, 4, 55, 50, 100)
	// b's last observation is 10h old: stale per the 3h rule, so its samples
	// are excluded from rates/ETA/projection but it counts toward total.
	s.Apply(AccountSnapshot{KeyID: "b", At: now.Add(-10 * time.Hour),
		Included: &USDWindow{Remaining: 30, Cap: 100}})
	view := BuildFleetView(s, []string{"a", "b"}, FleetOptions{
		Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{},
	}, nil)
	if view.TotalAccounts != 2 || view.ActiveAccounts != 1 || view.StaleAccounts != 1 {
		t.Fatalf("tallies = %+v", view)
	}
	if absDiff(view.RatePerHour, 1.25) > 0.02 {
		t.Fatalf("rate = %v, want a only (1.25)", view.RatePerHour)
	}
}

func TestFleetEmpty(t *testing.T) {
	now := time.Now().UTC()
	view := BuildFleetView(NewStore(24*time.Hour), nil, FleetOptions{
		Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{},
	}, nil)
	if view.TotalAccounts != 0 || view.Risk != RiskHealthy || view.FleetETA != 0 {
		t.Fatalf("empty fleet = %+v", view)
	}
	if len(view.Projection) != 48 {
		t.Fatalf("projection = %d points", len(view.Projection))
	}
}

func TestFleetWeeklyGateSuspendsBurn(t *testing.T) {
	// D7 coupling: weekly (30% left, 5%/h) empties at +6h; the shared
	// consump­tion stream then freezes the included window (40 left,
	// 5$/h) mid-drain at 10 USD until the weekly reset at +24h refills
	// the gate and the burn resumes.
	now := time.Now().UTC()
	s := NewStore(24 * time.Hour)
	reset := now.Add(24 * time.Hour)
	s.Apply(AccountSnapshot{KeyID: "p", Label: "pro", At: now.Add(-time.Hour),
		Included: &USDWindow{Remaining: 45, Cap: 100},
		Weekly:   &PctWindow{RemainingPercent: 35, ResetsAt: reset}})
	s.Apply(AccountSnapshot{KeyID: "p", Label: "pro", At: now,
		Included: &USDWindow{Remaining: 40, Cap: 100},
		Weekly:   &PctWindow{RemainingPercent: 30, ResetsAt: reset}})

	dl := now.Add(10 * time.Hour)
	view := BuildFleetView(s, []string{"p"}, FleetOptions{
		Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{},
	}, &dl)
	if view.AtDeadline == nil {
		t.Fatal("at_deadline missing")
	}
	// Included would drain to 0 at +8h on its own, but the weekly gate
	// empties first: capacity freezes at 10 USD and the account is dark.
	if absDiff(view.AtDeadline.AvailableUSD, 10) > 0.02 || view.AtDeadline.AvailableAccounts != 0 {
		t.Fatalf("at_deadline = %+v, want 10 USD / 0 accounts (frozen)", view.AtDeadline)
	}
	check := func(idx int, usd float64, active int) {
		pt := view.Projection[idx]
		if absDiff(pt[1], usd) > 0.02 || int(pt[2]) != active {
			t.Fatalf("projection[%d] = %v, want %v usd / %d active", idx, pt, usd, active)
		}
	}
	check(5, 10, 0)  // +6h: gate lands at 0 exactly; dark at the stamp, 10 USD frozen
	check(6, 10, 0)  // +7h: frozen
	check(23, 10, 0) // +24h: reset instant — bump condition needs a strictly later stamp
	check(24, 10, 1) // +25h: weekly refilled at the bump, burn resumes next step
	check(26, 0, 0)  // +27h: remaining 5 USD drained; included at empty → dark
	check(47, 0, 0)
}

func TestFleetETASkipsExhaustedAccounts(t *testing.T) {
	// Dark account (included drained, samples ≥3) contributes no capacity
	// and its leftover burn estimate must not shrink the fleet ETA: only
	// the live account (rem 50, 1.25$/h) drives 50/1.25 = 40h.
	now := time.Now().UTC()
	s := NewStore(24 * time.Hour)
	for i, rem := range []float64{10, 5, 0} {
		s.Apply(AccountSnapshot{KeyID: "a", At: now.Add(time.Duration(i-2) * time.Hour),
			Included: &USDWindow{Remaining: rem, Cap: 25}})
	}
	seedLinear(s, "b", now, 4, 55, 50, 100)
	view := BuildFleetView(s, []string{"a", "b"}, FleetOptions{
		Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{},
	}, nil)
	if view.ExhaustedAccounts != 1 {
		t.Fatalf("exhausted accounts = %d, want 1", view.ExhaustedAccounts)
	}
	if absDiff(view.FleetETA, 40*3600) > 60 {
		t.Fatalf("fleet eta = %v, want 40h (live account only)", view.FleetETA)
	}
	if absDiff(view.RemainingUSD, 50) > 0.01 {
		t.Fatalf("remaining = %v, want 50", view.RemainingUSD)
	}
}

func TestFleetRiskDarkFractionAxis(t *testing.T) {
	now := time.Now().UTC()
	opts := FleetOptions{Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{}}

	// Regression pin (D9): a Pro-shaped account — no included window,
	// purchased $0, healthy pct windows — must not fire the old
	// any-USD-zero critical trigger.
	s := NewStore(24 * time.Hour)
	for i, rem := range []float64{60, 40, 20} {
		s.Apply(AccountSnapshot{KeyID: "pro", At: now.Add(time.Duration(i-2) * time.Hour),
			Session: &PctWindow{RemainingPercent: rem}, PurchasedUSD: 0})
	}
	seedLinear(s, "max1", now, 4, 106, 102, 100) // 1$/h → horizon end 54 ≥ 30%
	view := BuildFleetView(s, []string{"pro", "max1"}, opts, nil)
	if view.Risk != RiskHealthy {
		t.Fatalf("pro-shaped fleet risk = %q, want healthy (purchased-$0 is not drained)", view.Risk)
	}

	// Dark axis: 1-of-5 waiting reset (20%) reads tight even when the
	// capacity stays untouched; zero dark reads healthy. (Cap 0 included
	// windows don't record at all, so the drained account carries a tiny
	// real allowance.)
	s = NewStore(24 * time.Hour)
	s.Apply(AccountSnapshot{KeyID: "d1", At: now.Add(-2 * time.Hour), Included: &USDWindow{Remaining: 2, Cap: 6}})
	s.Apply(AccountSnapshot{KeyID: "d1", At: now.Add(-time.Hour), Included: &USDWindow{Remaining: 1, Cap: 6}})
	s.Apply(AccountSnapshot{KeyID: "d1", At: now, Included: &USDWindow{Remaining: 0, Cap: 6}})
	for _, id := range []string{"h1", "h2", "h3", "h4"} {
		seedLinear(s, id, now, 4, 102, 100, 100) // 0.5$/h → horizon end 76
	}
	view = BuildFleetView(s, []string{"d1", "h1", "h2", "h3", "h4"}, opts, nil)
	if view.Risk != RiskTight {
		t.Fatalf("20%% dark risk = %q, want tight", view.Risk)
	}
	view = BuildFleetView(s, []string{"h1", "h2", "h3", "h4"}, opts, nil)
	if view.Risk != RiskHealthy {
		t.Fatalf("no-dark risk = %q, want healthy", view.Risk)
	}
}

func absDiff(a, b float64) float64 {
	d := a - b
	if d < 0 {
		return -d
	}
	return d
}
