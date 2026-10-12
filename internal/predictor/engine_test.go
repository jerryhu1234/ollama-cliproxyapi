package predictor

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func testFetchOK(remaining float64) FetchFn {
	return func(ctx context.Context, key string) (AccountSnapshot, error) {
		return AccountSnapshot{
			KeyID: "k1", At: time.Now().UTC(),
			Included: &USDWindow{Remaining: remaining, Cap: 100}, PurchasedUSD: -1,
		}, nil
	}
}

func fastOptions(fetch FetchFn) Options {
	return Options{
		Interval:        50 * time.Millisecond,
		BudgetPerMinute: 1000,
		History:         time.Hour,
		Horizon:         time.Hour,
		Keys:            []KeyRef{{ID: "k1", Label: "one", Key: "sk-test"}},
		Fetch:           fetch,
	}
}

func TestEngineSamplesAndJoins(t *testing.T) {
	var calls atomic.Int32
	fetch := func(ctx context.Context, key string) (AccountSnapshot, error) {
		calls.Add(1)
		return AccountSnapshot{
			KeyID: "k1", At: time.Now().UTC(),
			Included: &USDWindow{Remaining: 90 - float64(calls.Load()), Cap: 100}, PurchasedUSD: -1,
		}, nil
	}
	e := NewEngine(fastOptions(fetch))
	e.Start()
	time.Sleep(700 * time.Millisecond)
	e.Stop()
	if !e.WaitStopped(3 * time.Second) {
		t.Fatal("engine goroutines did not join after Stop")
	}
	series := e.Store().Series("k1", WindowIncluded)
	if len(series) < 2 {
		t.Fatalf("series = %d samples, want >= 2", len(series))
	}
	if int(calls.Load()) < len(series) {
		t.Fatalf("calls=%d but series=%d", calls.Load(), len(series))
	}
	if e.Store().Stale("k1", time.Minute) {
		t.Fatal("fresh samples marked stale")
	}
}

func TestEngine429BackoffSuppressesFetches(t *testing.T) {
	var calls atomic.Int32
	logMu := sync.Mutex{}
	logs := []map[string]any{}
	fetch := func(ctx context.Context, key string) (AccountSnapshot, error) {
		calls.Add(1)
		return AccountSnapshot{}, RetryAfter(errors.New("rate limited"), time.Hour)
	}
	opts := fastOptions(fetch)
	opts.Log = func(level, msg string, fields map[string]any) {
		logMu.Lock()
		defer logMu.Unlock()
		logs = append(logs, map[string]any{"level": level, "msg": msg, "fields": fields})
	}
	e := NewEngine(opts)
	e.Start()
	time.Sleep(500 * time.Millisecond)
	e.Stop()
	if !e.WaitStopped(3 * time.Second) {
		t.Fatal("engine did not join")
	}
	// The very first fetch arms a 1h budget backoff; every later round must
	// skip fetches until it elapses (plan §3: Retry-After wins, budget is
	// never burned against a rejecting upstream).
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1", got)
	}
	logMu.Lock()
	defer logMu.Unlock()
	found := false
	for _, l := range logs {
		if l["msg"] == "predictor: account sample failed" {
			fields := l["fields"].(map[string]any)
			if fields["backoff_seconds"] == 3600 {
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("backoff log missing: %+v", logs)
	}
}

func TestEngineRefreshSchedulesSample(t *testing.T) {
	var calls atomic.Int32
	fetch := func(ctx context.Context, key string) (AccountSnapshot, error) {
		calls.Add(1)
		return AccountSnapshot{KeyID: "k1", At: time.Now().UTC(),
			Included: &USDWindow{Remaining: 42, Cap: 100}, PurchasedUSD: -1}, nil
	}
	opts := Options{
		Interval:        10 * time.Minute, // refresh is the only way data lands here
		BudgetPerMinute: 200,
		History:         24 * time.Hour,
		Horizon:         time.Hour,
		Keys:            []KeyRef{{ID: "k1", Label: "one", Key: "sk-test"}},
		Fetch:           fetch,
	}
	e := NewEngine(opts)
	e.Start()
	if !e.RequestRefresh("k1") {
		t.Fatal("RequestRefresh rejected a configured key")
	}
	if e.RequestRefresh("ghost") {
		t.Fatal("RequestRefresh accepted an unknown key")
	}
	time.Sleep(1200 * time.Millisecond) // token drip: 60s/200 = 300ms
	e.Stop()
	if !e.WaitStopped(3 * time.Second) {
		t.Fatal("engine did not join")
	}
	if calls.Load() < 1 {
		t.Fatal("manual refresh did not produce an in-budget fetch")
	}
	series := e.Store().Series("k1", WindowIncluded)
	if len(series) < 1 || series[len(series)-1].Remaining != 42 {
		t.Fatalf("refreshed series = %+v", series)
	}
}

func TestEngineAccelerateDecision(t *testing.T) {
	e := NewEngine(Options{Interval: time.Minute, BudgetPerMinute: 8})
	now := time.Now().UTC()
	// Near exhaustion: last remaining 15 of cap 100 → fast lane.
	e.Store().Apply(AccountSnapshot{KeyID: "k1", At: now.Add(-2 * time.Hour),
		Included: &USDWindow{Remaining: 22, Cap: 100}})
	e.Store().Apply(AccountSnapshot{KeyID: "k1", At: now.Add(-time.Hour),
		Included: &USDWindow{Remaining: 18, Cap: 100}})
	e.Store().Apply(AccountSnapshot{KeyID: "k1", At: now,
		Included: &USDWindow{Remaining: 15, Cap: 100}})
	if !e.accelerate("k1") {
		t.Fatal("near-exhaustion account must accelerate")
	}
	// Healthy account: no trigger.
	e2 := NewEngine(Options{Interval: time.Minute, BudgetPerMinute: 8})
	e2.Store().Apply(AccountSnapshot{KeyID: "k2", At: now,
		Included: &USDWindow{Remaining: 90, Cap: 100}})
	if e2.accelerate("k2") {
		t.Fatal("healthy account must not accelerate")
	}
	// Burn-rate spike: 19h flat then 6h at 6× → cons > 2×opt.
	e3 := NewEngine(Options{Interval: time.Minute, BudgetPerMinute: 8})
	rem := 120.0
	for i := 0; i < 25; i++ {
		at := now.Add(time.Duration(i-25) * time.Hour)
		burn := 1.0
		if i >= 19 {
			burn = 6
		}
		rem -= burn
		e3.Store().Apply(AccountSnapshot{KeyID: "k3", At: at,
			Included: &USDWindow{Remaining: rem, Cap: 100}})
	}
	if !e3.accelerate("k3") {
		t.Fatal("sudden burn spike must accelerate")
	}
}

func TestSnapshotShape(t *testing.T) {
	e := NewEngine(fastOptions(testFetchOK(90)))
	now := time.Now().UTC()
	e.Store().Apply(AccountSnapshot{KeyID: "k1", Label: "one", At: now.Add(-time.Hour),
		Included: &USDWindow{Remaining: 91, Cap: 100}, PurchasedUSD: -1})
	e.Store().Apply(AccountSnapshot{KeyID: "k1", Label: "one", At: now,
		Included: &USDWindow{Remaining: 90, Cap: 100}, PurchasedUSD: -1})

	data := e.Snapshot(nil)
	if data.GeneratedAt == "" || len(data.Accounts) != 1 {
		t.Fatalf("response = %+v", data)
	}
	acc := data.Accounts[0]
	if acc.KeyID != "k1" || acc.Stale {
		t.Fatalf("account = %+v", acc)
	}
	if len(acc.Windows) != 1 || acc.Windows[0].Name != WindowIncluded {
		t.Fatalf("windows = %+v", acc.Windows)
	}
	if acc.Windows[0].State != StateNoData { // only two samples
		t.Fatalf("state = %q", acc.Windows[0].State)
	}
	fleet := data.ByType["ollama"]
	if fleet.TotalAccounts != 1 || fleet.ActiveAccounts != 1 {
		t.Fatalf("fleet = %+v", fleet)
	}
	// Deadline path: evaluation card appears when asked.
	dl := now.Add(5 * time.Hour)
	data = e.Snapshot(&dl)
	if data.ByType["ollama"].AtDeadline == nil {
		t.Fatal("deadline evaluation missing")
	}
}
