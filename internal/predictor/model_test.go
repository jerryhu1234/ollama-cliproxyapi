package predictor

import (
	"math"
	"testing"
	"time"
)

// ---- synthetic series helpers ---------------------------------------------

// linearSeries walks remaining linearly: remStart - burn * step * i.
func linearSeries(start time.Time, count int, remStart, burn float64, step time.Duration) []WindowSample {
	out := make([]WindowSample, 0, count)
	rem := remStart
	for i := 0; i < count; i++ {
		out = append(out, WindowSample{Ts: start.Add(time.Duration(i) * step), Remaining: rem, Cap: 100})
		rem -= burn * step.Hours()
	}
	return out
}

// ---- RateSource -------------------------------------------------------------

func TestEstimateLinearBurn(t *testing.T) {
	base := time.Date(2026, 10, 9, 6, 0, 0, 0, time.UTC)
	series := linearSeries(base, 25, 50, 2, time.Hour) // 25h at -2/h
	src := SamplingRateSource{}
	cons, opt, r2, samples := src.Estimate(UnitUSD, series, base.Add(24*time.Hour))
	if samples != 25 {
		t.Fatalf("samples = %d", samples)
	}
	if math.Abs(cons-2) > 0.01 || math.Abs(opt-2) > 0.01 || r2 < 0.99 {
		t.Fatalf("cons=%v opt=%v r2=%v", cons, opt, r2)
	}
}

func TestEstimateShortSpikeResponsive(t *testing.T) {
	// 18h at 1$/h then 6h at 6$/h: the short window must track the spike
	// (responsive) while the long window averages (stabilizer) — the D6
	// conservative=fast, optimistic=slow pairing.
	base := time.Date(2026, 10, 9, 0, 0, 0, 0, time.UTC)
	rem := 90.0
	var series []WindowSample
	for i := 0; i < 24; i++ {
		rem -= 1
		if i >= 18 {
			rem -= 5 // total burn 6 in the spike hours
		}
		series = append(series, WindowSample{Ts: base.Add(time.Duration(i+1) * time.Hour), Remaining: rem, Cap: 100})
	}
	src := SamplingRateSource{}
	cons, opt, _, _ := src.Estimate(UnitUSD, series, base.Add(24*time.Hour))
	if math.Abs(cons-6) > 0.35 {
		t.Fatalf("cons = %v, want ≈6 (short window tracks the spike)", cons)
	}
	if opt < 1.7 || opt > 2.7 {
		t.Fatalf("opt = %v, want ≈1.9 (long window averages)", opt)
	}
	if !(cons > 2*opt) {
		t.Fatalf("cons(%v) must dominate 2×opt(%v) in a spike", cons, opt)
	}
}

func TestCurrentSegmentSplitsOnResetBump(t *testing.T) {
	base := time.Date(2026, 10, 9, 0, 0, 0, 0, time.UTC)
	series := []WindowSample{
		{Ts: base, Remaining: 3, Cap: 100},
		{Ts: base.Add(time.Hour), Remaining: 2, Cap: 100},
		{Ts: base.Add(2 * time.Hour), Remaining: 99, Cap: 100}, // reset bump
		{Ts: base.Add(3 * time.Hour), Remaining: 98, Cap: 100},
	}
	seg := currentSegment(UnitUSD, series, base.Add(3*time.Hour))
	if len(seg) != 2 || seg[0].Remaining != 99 {
		t.Fatalf("seg = %+v", seg)
	}
	// Percent windows: tolerance 5 points, no cap.
	pct := []WindowSample{
		{Ts: base, Remaining: 9},
		{Ts: base.Add(time.Hour), Remaining: 80},
	}
	segPct := currentSegment(UnitPct, pct, base.Add(time.Hour))
	if len(segPct) != 1 || segPct[0].Remaining != 80 {
		t.Fatalf("pct seg = %+v", segPct)
	}
}

// ---- shape-aware eta walk ----------------------------------------------------

func profileHeavyEvening() map[int]float64 {
	p := make(map[int]float64)
	for h := 0; h < 24; h++ {
		p[h] = 1.0 / 3
	}
	p[19] = 2.0
	p[20] = 2.5
	p[21] = 2.0
	// mass = 21·(1/3) + 6.5 = 13.5 → scale = 24/13.5 ≈ 1.778; the walk
	// averages ~0.59/h on plain hours and 3.56/4.44/3.56 on the peak.
	return p
}

func TestWalkEtaFlat(t *testing.T) {
	base := time.Date(2026, 10, 9, 6, 0, 0, 0, time.UTC)
	eta, burn, exhausts := walkEta(30, 2, nil, base, time.Time{}, etaWalkMaxHour)
	if !exhausts || burn != 30 || eta != 15*time.Hour {
		t.Fatalf("flat walk = %v %v %t", burn, eta, exhausts)
	}
}

func TestWalkEtaResetsBound(t *testing.T) {
	base := time.Date(2026, 10, 9, 6, 0, 0, 0, time.UTC)
	eta, burn, exhausts := walkEta(30, 2, nil, base, base.Add(5*time.Hour), etaWalkMaxHour)
	if exhausts || eta != 5*time.Hour || burn != 10 {
		t.Fatalf("bounded walk = eta %v burn %v exhausts %t", eta, burn, exhausts)
	}
}

func TestWalkEtaShapeAware(t *testing.T) {
	// Anchor the walk at 17:00 UTC so hours 18-21 hit the evening peak.
	// Cumulative burn by 24:00 is 13.33 of the 20 remaining; the plain-hour
	// pace (0.593/h) then drains the rest in ≈10.25h → ETA ≈17.25h, i.e.
	// the peak pulled exhaustion earlier than the flat 20h without changing
	// the total burn (mass conservation below).
	base := time.Date(2026, 10, 9, 17, 0, 0, 0, time.UTC)
	eta, burn, exhausts := walkEta(20, 1, profileHeavyEvening(), base, time.Time{}, etaWalkMaxHour)
	if !exhausts {
		t.Fatal("shape walk must exhaust")
	}
	if math.Abs(burn-20) > 0.01 {
		t.Fatalf("burn = %v, want 20 (mass conservation)", burn)
	}
	if eta <= 10*time.Hour || eta >= 20*time.Hour {
		t.Fatalf("eta = %v, want strictly inside (10h, 20h) — peak pulls it early", eta)
	}
	// Degenerate profile (all zeros) degrades to flat.
	etaFlat, _, exhaustsFlat := walkEta(20, 1, map[int]float64{19: 0, 20: 0}, base, time.Time{}, etaWalkMaxHour)
	if !exhaustsFlat || etaFlat != 20*time.Hour {
		t.Fatalf("degenerate profile fell back to %v, want flat 20h", etaFlat)
	}
}

// ---- state machine & confidence ----------------------------------------------

func TestWindowViewStates(t *testing.T) {
	base := time.Date(2026, 10, 9, 0, 0, 0, 0, time.UTC)
	now := base.Add(10 * time.Hour)
	src := SamplingRateSource{}

	burning := linearSeries(base, 11, 20, 1, time.Hour) // rem now = 10
	v := buildWindowView(WindowIncluded, burning, now, nil, src, false)
	if v.State != StateExhaustedIn || v.Eta < 9.5*3600 || v.Eta > 10.5*3600 {
		t.Fatalf("burning view = %+v", v)
	}
	if v.Confidence != ConfidenceHigh { // 11 samples, r2≈1
		t.Fatalf("confidence = %q", v.Confidence)
	}
	if v.RatePerHour != 1 || v.OptimisticPH != 1 {
		t.Fatalf("rates = %v/%v", v.RatePerHour, v.OptimisticPH)
	}

	// Same window but resetting in 3h: resets-first with burn-to-reset 3.
	resetting := make([]WindowSample, len(burning))
	copy(resetting, burning)
	for i := range resetting {
		resetting[i].ResetsAt = now.Add(3 * time.Hour)
	}
	v = buildWindowView(WindowIncluded, resetting, now, nil, src, false)
	if v.State != StateResetsFirst || v.BurnToReset != 3 {
		t.Fatalf("reset-first view = %+v", v)
	}

	flat := linearSeries(base, 11, 50, 0, time.Hour)
	v = buildWindowView(WindowIncluded, flat, now, nil, src, false)
	if v.State != StateStable {
		t.Fatalf("stable view = %+v", v)
	}

	short := linearSeries(base, 2, 50, 1, time.Hour)
	v = buildWindowView(WindowIncluded, short, now, nil, src, false)
	if v.State != StateNoData || v.Confidence != ConfidenceLow {
		t.Fatalf("no-data view = %+v", v)
	}

	exhausted := linearSeries(base, 11, 0, 1, time.Hour)
	v = buildWindowView(WindowIncluded, exhausted, now, nil, src, false)
	if v.State != StateExhausted {
		t.Fatalf("exhausted view = %+v", v)
	}

	empty := buildWindowView(WindowIncluded, nil, now, nil, src, false)
	if empty.State != StateNoData || empty.Eta != 0 {
		t.Fatalf("empty view = %+v", empty)
	}
}

func TestWindowViewUnitSemantics(t *testing.T) {
	base := time.Date(2026, 10, 9, 0, 0, 0, 0, time.UTC)
	now := base.Add(5 * time.Hour)
	// Legacy session window burns in percent points; profile must not apply.
	series := linearSeries(base, 6, 100, 4, time.Hour) // rem now = 80
	for i := range series {
		series[i].Cap = 0
	}
	v := buildWindowView(WindowSession, series, now, profileHeavyEvening(), SamplingRateSource{}, false)
	if v.Unit != UnitPct || v.ShapeAware {
		t.Fatalf("pct view unit/shape = %q/%t", v.Unit, v.ShapeAware)
	}
	if v.State != StateExhaustedIn || math.Abs(v.Eta-20*3600) > 3600 {
		t.Fatalf("pct view = %+v (want 20h: 80 points at 4 points/h)", v)
	}
}

func TestViewGatingAnnotation(t *testing.T) {
	now := time.Date(2026, 10, 9, 10, 0, 0, 0, time.UTC)
	session := func() WindowView {
		return WindowView{Name: WindowSession, Unit: UnitPct, State: StateExhaustedIn, Eta: 10 * 3600}
	}
	weekly := func() WindowView {
		return WindowView{Name: WindowWeekly, Unit: UnitPct, State: StateExhaustedIn, Eta: 4 * 3600}
	}

	// Weekly drains first and never refills before the session would run
	// out → the session window is frozen by the weekly gate from +4h.
	wins := []WindowView{session(), weekly()}
	applyGating(wins, now)
	if wins[0].GatedBy != WindowWeekly {
		t.Fatalf("session gating = %+v", wins[0])
	}
	if at, err := time.Parse(time.RFC3339, wins[0].GatedAt); err != nil || !at.Equal(now.Add(4*time.Hour)) {
		t.Fatalf("gated_at = %q (err %v)", wins[0].GatedAt, err)
	}
	if wins[1].GatedBy != "" {
		t.Fatalf("gate itself must not be annotated: %+v", wins[1])
	}

	// Refill lands before the session's depletion: standby pause only,
	// keep the standalone eta.
	wins = []WindowView{session(), weekly()}
	wins[1].ResetsAt = FormatTimestamp(now.Add(8 * time.Hour))
	applyGating(wins, now)
	if wins[0].GatedBy != "" {
		t.Fatalf("refill-in-time must not annotate: %+v", wins[0])
	}

	// Already-empty gate without a refill (blank resets): freezes from now.
	wins = []WindowView{session(), weekly()}
	wins[1].State = StateExhausted
	applyGating(wins, now)
	if wins[0].GatedBy != WindowWeekly || wins[0].GatedAt != FormatTimestamp(now) {
		t.Fatalf("exhausted gate annotation = %+v", wins[0])
	}

	// Included is a serving gate, not a coupling gate: exhaustion of the
	// USD allowance does not freeze the session meter (billing continues).
	wins = []WindowView{session(), {Name: WindowIncluded, Unit: UnitUSD, State: StateExhaustedIn, Eta: 2 * 3600}}
	applyGating(wins, now)
	if wins[0].GatedBy != "" {
		t.Fatalf("included must not gate the session: %+v", wins[0])
	}
}

func TestAccountStatusDerivation(t *testing.T) {
	now := time.Date(2026, 10, 9, 10, 0, 0, 0, time.UTC)
	src := SamplingRateSource{}
	view := func(s *Store, id string) AccountView {
		return buildAccountView(s, id, now, 3*time.Hour, src, nil)
	}

	// Pct hard gate at empty → exhausted, reporting the gate's reset.
	s := NewStore(24 * time.Hour)
	reset := now.Add(2 * time.Hour)
	for i, rem := range []float64{40, 20, 0} {
		s.Apply(AccountSnapshot{KeyID: "p", At: now.Add(time.Duration(i-2) * time.Hour),
			Session: &PctWindow{RemainingPercent: rem, ResetsAt: reset}})
	}
	av := view(s, "p")
	if av.Status != StatusExhausted {
		t.Fatalf("pct-empty status = %q", av.Status)
	}
	if at, err := time.Parse(time.RFC3339, av.NextReset); err != nil || !at.Equal(reset) {
		t.Fatalf("next_reset = %q (err %v)", av.NextReset, err)
	}

	// Included serving window at empty (no pct gates) → exhausted.
	s = NewStore(24 * time.Hour)
	for i, rem := range []float64{10, 5, 0} {
		s.Apply(AccountSnapshot{KeyID: "u", At: now.Add(time.Duration(i-2) * time.Hour),
			Included: &USDWindow{Remaining: rem, Cap: 25}})
	}
	if av = view(s, "u"); av.Status != StatusExhausted {
		t.Fatalf("included-empty status = %q", av.Status)
	}

	// Purchased drains to zero freely — billing channel, never a gate.
	s = NewStore(24 * time.Hour)
	for i := 0; i < 3; i++ {
		s.Apply(AccountSnapshot{KeyID: "x", At: now.Add(time.Duration(i-2) * time.Hour), PurchasedUSD: 0})
	}
	if av = view(s, "x"); av.Status != StatusReady {
		t.Fatalf("purchased-only status = %q", av.Status)
	}

	// Below the 3-sample no-data threshold even a zero reading cannot
	// gate the account.
	s = NewStore(24 * time.Hour)
	for i, rem := range []float64{5, 0} {
		s.Apply(AccountSnapshot{KeyID: "n", At: now.Add(time.Duration(i-1) * time.Hour),
			Session: &PctWindow{RemainingPercent: rem}})
	}
	if av = view(s, "n"); av.Status != StatusReady {
		t.Fatalf("no-data-gated status = %q", av.Status)
	}

	// Healthy roomy windows → ready.
	s = NewStore(24 * time.Hour)
	for i, rem := range []float64{50, 40, 30} {
		s.Apply(AccountSnapshot{KeyID: "h", At: now.Add(time.Duration(i-2) * time.Hour),
			Included: &USDWindow{Remaining: rem, Cap: 100}})
	}
	if av = view(s, "h"); av.Status != StatusReady {
		t.Fatalf("healthy status = %q", av.Status)
	}
}

func TestConfidenceLadder(t *testing.T) {
	cases := []struct {
		samples int
		r2      float64
		stable  bool
		want    string
	}{
		{9, 0.9, false, ConfidenceHigh},
		{12, 0.5, true, ConfidenceHigh},
		{4, 0.6, false, ConfidenceMedium},
		{3, 0.3, false, ConfidenceLow},
		{2, 0.99, false, ConfidenceLow},
	}
	for _, tc := range cases {
		if got := classifyConfidence(tc.samples, tc.r2, tc.stable); got != tc.want {
			t.Fatalf("classify(%d, %.2f, %t) = %q, want %q", tc.samples, tc.r2, tc.stable, got, tc.want)
		}
	}
}
