package predictor

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Options wire the engine to the plugin world. Fetch is the only upstream
// touchpoint; Log and Now are injectable for tests.
type Options struct {
	Interval        time.Duration // base sampling cadence (plan 10m)
	BudgetPerMinute int           // global upstream call budget (plan 8)
	History         time.Duration // retained series span (plan 7d)
	Horizon         time.Duration // fleet projection span (plan 48h)
	StaleAfter      time.Duration // samples older than this are stale (plan 3×)
	DataDir         string        // "" = memory-only persistence
	Keys            []KeyRef
	Fetch           FetchFn
	Now             func() time.Time
	Log             func(level, msg string, fields map[string]any)
}

// HistoryOrDefault applies the package default when unset.
func (o Options) HistoryOrDefault() time.Duration {
	if o.History > 0 {
		return o.History
	}
	return DefaultHistory
}

// fetchTimeout bounds one upstream observation (management surface pace).
const fetchTimeout = 45 * time.Second

// flushInterval rides the plan §7 policy: append every 5 minutes + one
// final flush at shutdown; a crash loses at most 5 minutes of samples.
const flushInterval = 5 * time.Minute

// Engine owns the sampler lifecycle, the store, and view assembly. All
// public entry points (Snapshot / RequestRefresh / Stop) are goroutine-safe.
type Engine struct {
	opts  Options
	store *Store
	perse *persister

	ctx      context.Context
	stop     chan struct{}
	stopOnce sync.Once
	done     chan struct{}
	wg       sync.WaitGroup

	tokens  chan struct{} // global call budget
	refresh map[string]chan struct{}

	backMu    sync.Mutex
	backoffs  map[string]time.Time
	failCount map[string]int
	jrngMu    sync.Mutex
	jrng      *rand.Rand

	persistMu sync.Mutex
	persist   PersistStatus
}

// Persistence states surfaced to the page (plan §13 D10): degradation is
// visible — the D1 "restarts keep history" promise becomes verifiable.
const (
	PersistOK           = "ok"           // history restored / saving normally
	PersistOff          = "off"          // no data dir (config/"off" or home resolution)
	PersistStateLost    = "load-failed"  // history file could not be restored
	PersistStateUnsaved = "write-failed" // saving stopped; file keeps prior data
)

// PersistStatus is the durable-layer health in the page payload.
type PersistStatus struct {
	State  string `json:"state"`
	Detail string `json:"detail,omitempty"`
}

func (e *Engine) setPersist(state, detail string) {
	e.persistMu.Lock()
	defer e.persistMu.Unlock()
	e.persist = PersistStatus{State: state, Detail: detail}
}

// PersistStatus reports the durable-layer health (Snapshot/page only).
func (e *Engine) PersistStatus() PersistStatus {
	e.persistMu.Lock()
	defer e.persistMu.Unlock()
	return e.persist
}

// persister appends NDJSON records to dataDir/usage-samples.ndjson with a
// load-then-compact startup (plan §7).
type persister struct {
	path string
	mu   sync.Mutex
}

// NewEngine validates options and returns an engine. Persistence failure
// (bad dir) is logged once by Start; the engine degrades to memory-only.
func NewEngine(opts Options) *Engine {
	if opts.Interval <= 0 {
		opts.Interval = DefaultInterval
	}
	if opts.BudgetPerMinute <= 0 {
		opts.BudgetPerMinute = DefaultBudget
	}
	if opts.Horizon <= 0 {
		opts.Horizon = DefaultHorizon
	}
	if opts.StaleAfter <= 0 {
		opts.StaleAfter = 3 * opts.Interval
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.Log == nil {
		opts.Log = func(level, msg string, fields map[string]any) {}
	}
	e := &Engine{
		opts:      opts,
		store:     NewStore(opts.HistoryOrDefault()),
		stop:      make(chan struct{}),
		done:      make(chan struct{}),
		tokens:    make(chan struct{}, opts.BudgetPerMinute),
		refresh:   make(map[string]chan struct{}, len(opts.Keys)),
		backoffs:  make(map[string]time.Time),
		failCount: make(map[string]int),
		jrng:      rand.New(rand.NewPCG(uint64(opts.Now().UnixNano()), 0x011ada7a)),
	}
	for _, kr := range opts.Keys {
		e.refresh[kr.ID] = make(chan struct{}, 1)
	}
	if opts.DataDir != "" {
		e.perse = &persister{path: filepath.Join(opts.DataDir, "usage-samples.ndjson")}
	}
	return e
}

// Start boots the stop-context, loads persisted history (compacting the
// file), and launches refiller, flusher, and per-key sampler loops.
// First observations are staggered so accounts never collide on the
// shared upstream budget.
func (e *Engine) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	e.ctx = ctx
	go func() {
		select {
		case <-e.stop:
			cancel()
		case <-ctx.Done():
		}
	}()
	if e.perse == nil {
		e.setPersist(PersistOff, "no data dir")
	} else {
		e.restorePersisted()
	}
	e.wg.Add(2)
	go e.refillLoop()
	go e.flushLoop()
	// Phase = stagger*(i+1): the FIRST sample of every key also waits a
	// full stagger slot, so a lifecycle (register/reconfigure) never fires
	// upstream calls synchronously — the host sees its first predictor
	// traffic no earlier than one sampling interval in, and the manual
	// refresh path is the immediate-data escape hatch (plan §6).
	stagger := e.opts.Interval / time.Duration(max(1, len(e.opts.Keys)))
	for i, kr := range e.opts.Keys {
		kr := kr
		e.wg.Add(1)
		go e.keyLoop(kr, stagger*time.Duration(i+1))
	}
	go func() {
		e.wg.Wait()
		e.finalFlush()
		close(e.done)
	}()
}

// restorePersisted streams the history file into the store line by line
// (plan §13 D10): the old monolithic read with its 64MiB cap rejected
// oversized files outright and degraded to memory-only forever — once the
// file crossed the cap, every restart silently dropped all history and the
// compaction that could have shrunk the file was unreachable (ratchet).
// Memory use now stays bounded by retention after a single trim pass, so
// any file size restores as cheaply as a fresh one.
func (e *Engine) restorePersisted() {
	n, err := e.perse.load(func(rec persistRecord) { e.store.IngestRecord(rec) })
	switch {
	case err != nil:
		e.setPersist(PersistStateLost, err.Error())
		e.opts.Log("warn", "predictor: persistence load failed, continuing memory-only",
			map[string]any{"error": err.Error()})
		e.perse = nil // never rewrite a file we failed to read
		return
	case n == 0:
		e.setPersist(PersistOK, "") // fresh boot: nothing to restore
		return
	}
	e.store.TrimRetention(e.now())
	// Compact: rewrite from retained state so the file stops growing
	// across restarts (plan §7).
	if err := e.perse.rewrite(e.store.SnapshotRecords()); err != nil {
		e.setPersist(PersistStateUnsaved, err.Error())
		e.opts.Log("warn", "predictor: persistence compaction failed",
			map[string]any{"error": err.Error()})
		e.perse = nil
	} else {
		e.setPersist(PersistOK, "")
	}
	e.opts.Log("info", "predictor: restored sampled history", map[string]any{
		"records": n, "keys": len(e.store.Keys()),
	})
}

// Stop cancels in-flight fetches and joins the loops. WaitStopped gives
// the joined-handle wait for the caller's bounded drain.
func (e *Engine) Stop() {
	e.stopOnce.Do(func() { close(e.stop) })
}

// WaitStopped waits up to timeout for the engine's goroutines to join.
func (e *Engine) WaitStopped(timeout time.Duration) bool {
	select {
	case <-e.done:
		return true
	case <-time.After(timeout):
		return false
	}
}

// Store exposes the engine's series store (views/tests only).
func (e *Engine) Store() *Store { return e.store }

func (e *Engine) now() time.Time { return e.opts.Now().UTC() }

func (e *Engine) stopCtx() (context.Context, context.CancelFunc) {
	return context.WithTimeout(e.ctx, fetchTimeout)
}

// refillLoop drips the global call budget into the token channel.
func (e *Engine) refillLoop() {
	defer e.wg.Done()
	interval := time.Minute / time.Duration(max(1, e.opts.BudgetPerMinute))
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-e.stop:
			return
		case <-t.C:
			select {
			case e.tokens <- struct{}{}:
			default:
			}
		}
	}
}

// flushLoop drains the store's dirty records onto disk every 5 minutes.
func (e *Engine) flushLoop() {
	defer e.wg.Done()
	t := time.NewTicker(flushInterval)
	defer t.Stop()
	for {
		select {
		case <-e.stop:
			return
		case <-t.C:
			e.flush()
		}
	}
}

func (e *Engine) flush() {
	if e.perse == nil {
		return
	}
	recs := e.store.DrainDirty()
	if len(recs) > 0 {
		if err := e.perse.append(recs); err != nil {
			e.opts.Log("warn", "predictor: persistence append failed",
				map[string]any{"error": err.Error()})
			if st := e.PersistStatus(); st.State == PersistOK {
				e.setPersist(PersistStateUnsaved, err.Error())
			}
		} else if st := e.PersistStatus(); st.State != PersistOK {
			e.setPersist(PersistOK, "") // recovered after a failed append
		}
	}
	e.compactIfLarge()
}

// compactIfLarge rewrites the history file from retained state once it
// outgrows persistCompactBytes (plan §13 D10: steady-state compaction —
// file growth self-limits without depending on restarts).
func (e *Engine) compactIfLarge() {
	info, err := os.Stat(e.perse.path)
	if err != nil || info.Size() <= persistCompactBytes {
		return
	}
	if err := e.perse.rewrite(e.store.SnapshotRecords()); err != nil {
		e.opts.Log("warn", "predictor: history compaction failed",
			map[string]any{"error": err.Error()})
	} else {
		e.opts.Log("info", "predictor: compacted history file",
			map[string]any{"bytes_before": info.Size()})
	}
}

// finalFlush runs after every loop joined — the shutdown-time flush.
func (e *Engine) finalFlush() { e.flush() }

// keyLoop drives one credential's sampling: staggered start, jittered
// cadence, refresh-boost, and adaptive acceleration (plan §3/D3).
func (e *Engine) keyLoop(kr KeyRef, phase time.Duration) {
	defer e.wg.Done()
	t := time.NewTimer(phase)
	defer t.Stop()
	for {
		var interval time.Duration
		select {
		case <-e.stop:
			return
		case <-e.refresh[kr.ID]:
			interval = e.sampleKey(kr) // manual boost: sample immediately
		case <-t.C:
			interval = e.sampleKey(kr)
		}
		e.jrngMu.Lock()
		base := time.Duration(float64(interval) * (0.8 + 0.4*e.jrng.Float64()))
		e.jrngMu.Unlock()
		if base < 250*time.Millisecond {
			base = 250 * time.Millisecond // loop floor; real cadence is config-validated >= 2m
		}
		if !t.Stop() {
			select {
			case <-t.C:
			default:
			}
		}
		t.Reset(base)
	}
}

// sampleKey performs one observation for a key and returns the cadence for
// the next round (base or accelerated interval, plan D3). A key waiting in
// 429-backoff returns early with the base interval.
func (e *Engine) sampleKey(kr KeyRef) time.Duration {
	base := e.opts.Interval
	select {
	case <-e.stop:
		return base
	case <-e.tokens:
	}
	e.backMu.Lock()
	until, wait := e.backoffs[kr.ID]
	e.backMu.Unlock()
	if wait && e.now().Before(until) {
		return base
	}
	if e.opts.Fetch == nil {
		return base
	}
	ctx, cancel := e.stopCtx()
	defer cancel()
	snap, err := e.opts.Fetch(ctx, kr.Key)
	if err != nil {
		e.recordFailure(kr, err)
		return base
	}
	e.backMu.Lock()
	delete(e.backoffs, kr.ID)
	e.failCount[kr.ID] = 0
	e.backMu.Unlock()
	if snap.KeyID == "" {
		snap.KeyID = kr.ID
	}
	if snap.Label == "" {
		snap.Label = kr.Label
	}
	if snap.At.IsZero() {
		snap.At = e.now()
	}
	e.store.Apply(snap)
	if e.accelerate(kr.ID) {
		return e.accelInterval(base)
	}
	return base
}

// accelerate reports whether a key enters the near-exhaustion fast lane:
// any window under 20% of cap (or percent), or short-term burn twice the
// long-window rate (plan D3: precision matters most close to exhaustion).
func (e *Engine) accelerate(keyID string) bool {
	src := SamplingRateSource{}
	for _, name := range e.store.WindowNames(keyID) {
		series := e.store.Series(keyID, name)
		if len(series) == 0 {
			continue
		}
		unit := windowUnit(name)
		last := series[len(series)-1]
		if unit == UnitUSD && last.Cap > 0 && last.Remaining < 0.20*last.Cap {
			return true
		}
		if unit == UnitPct && last.Remaining < 20 {
			return true
		}
		cons, opt, _, samples := src.Estimate(unit, series, e.now())
		if samples >= 4 && cons > epsUSDRate && opt > 0 && cons > 2*opt {
			return true
		}
	}
	return false
}

func (e *Engine) accelInterval(base time.Duration) time.Duration {
	half := base / 2
	if half < 2*time.Minute {
		half = 2 * time.Minute
	}
	return half
}

// recordFailure logs a redacted failure and arms 429 backoff per plan §3.
func (e *Engine) recordFailure(kr KeyRef, err error) {
	e.backMu.Lock()
	e.failCount[kr.ID]++
	n := e.failCount[kr.ID]
	e.backMu.Unlock()
	fields := map[string]any{"key": kr.ID, "count": n, "error": err.Error()}
	if ra, ok := retryAfter(err); ok && ra > 0 {
		e.backMu.Lock()
		e.backoffs[kr.ID] = e.now().Add(ra)
		e.backMu.Unlock()
		fields["backoff_seconds"] = int(ra.Seconds())
	}
	// First failure warns immediately; a storm degrades to every 12th so a
	// long outage cannot flood the host log.
	if n == 1 || n%12 == 0 {
		e.opts.Log("warn", "predictor: account sample failed", fields)
	}
}

// retryAfterError annotates upstream 429s with their Retry-After hint.
type retryAfterError struct {
	Seconds time.Duration
	Err     error
}

func (r *retryAfterError) Error() string { return r.Err.Error() }
func (r *retryAfterError) Unwrap() error { return r.Err }

// RetryAfter wraps err with an upstream retry hint (seconds >= 0).
func RetryAfter(err error, seconds time.Duration) error {
	return &retryAfterError{Seconds: seconds, Err: err}
}

func retryAfter(err error) (time.Duration, bool) {
	var ra *retryAfterError
	if errors.As(err, &ra) {
		return ra.Seconds, true
	}
	return 0, false
}

// RequestRefresh schedules an out-of-band observation for one key inside
// the same token budget (plan §6 manual refresh). Non-blocking.
func (e *Engine) RequestRefresh(keyID string) bool {
	ch, ok := e.refresh[keyID]
	if !ok {
		return false
	}
	select {
	case ch <- struct{}{}:
		return true
	default:
		return true // one already queued: counts as accepted
	}
}

// Snapshot assembles the full page/API response (plan §6): account cards
// for every configured key plus the ollama fleet aggregate. Reads are
// store-only — zero upstream calls.
func (e *Engine) Snapshot(deadline *time.Time) Response {
	now := e.now()
	resp := Response{GeneratedAt: FormatTimestamp(now), Accounts: make([]AccountView, 0, len(e.opts.Keys))}
	accel := make(map[string]bool, len(e.opts.Keys))
	keyIDs := make([]string, 0, len(e.opts.Keys))
	for _, kr := range e.opts.Keys {
		keyIDs = append(keyIDs, kr.ID)
		accel[kr.ID] = e.accelerate(kr.ID)
		resp.Accounts = append(resp.Accounts, buildAccountView(
			e.store, kr.ID, now, e.opts.StaleAfter, SamplingRateSource{}, accel,
		))
	}
	resp.ByType = map[string]FleetView{
		"ollama": BuildFleetView(e.store, keyIDs, FleetOptions{
			Now: now, Horizon: e.opts.Horizon, StaleAfter: e.opts.StaleAfter,
			Src: SamplingRateSource{},
		}, deadline),
	}
	// D10 visibility: report the RECORDED durable status, never infer it
	// from e.perse — a load failure nils the persister while the page must
	// still distinguish "load-failed" from "off".
	if st := e.PersistStatus(); st.State != PersistOK && st.State != "" {
		resp.Persistence = &st
	}
	return resp
}

// Response is the prediction API payload (plan §6). Persistence is nil
// only in the disabled-engine placeholder response.
type Response struct {
	GeneratedAt string               `json:"generated_at"`
	Persistence *PersistStatus       `json:"persistence,omitempty"`
	Accounts    []AccountView        `json:"accounts"`
	ByType      map[string]FleetView `json:"by_type"`
}

// persistCompactBytes is the steady-state compaction threshold (plan §13
// D10): once the history file outgrows this, flush rewrites it from
// retained state. Growth self-limits — a bug producing 60× write
// amplification can no longer run the file into the gigabytes. A var so
// tests can shrink the threshold.
var persistCompactBytes = int64(8 << 20)

// load streams the history file into sink line by line (plan §13 D10):
// no monolithic read and no size cap. A torn trailing line from a crash
// decodes as skipped; any oversized file restores as cheaply as a fresh
// one because records flow straight into the store (bounded by retention
// at the trim pass).
func (p *persister) load(sink func(persistRecord)) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	f, err := os.Open(p.path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil // fresh boot: nothing to restore
		}
		return 0, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	n := 0
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		rec, ok := DecodeRecord(line)
		if !ok {
			continue // torn line after a crash: skip and keep going
		}
		sink(rec)
		n++
	}
	return n, scanner.Err()
}

func (p *persister) append(recs []persistRecord) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(p.path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(p.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(EncodeRecords(recs))
	return err
}

// rewrite replaces the file atomically (tmp + rename) — the compaction
// path, also used right after a restore so the file stops growing.
func (p *persister) rewrite(recs []persistRecord) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(p.path), 0o755); err != nil {
		return err
	}
	tmp := p.path + ".tmp"
	if err := os.WriteFile(tmp, EncodeRecords(recs), 0o600); err != nil {
		_ = os.Remove(tmp) // D10: no stray half-written temp files left behind
		return err
	}
	if err := os.Rename(tmp, p.path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// Package-level defaults mirroring config (kept local so the predictor has
// no import cycle into internal/config).
const (
	DefaultInterval = 10 * time.Minute
	DefaultBudget   = 8
	DefaultHorizon  = 48 * time.Hour
)
