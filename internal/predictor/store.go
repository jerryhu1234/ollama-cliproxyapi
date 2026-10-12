package predictor

import (
	"sync"
	"time"
)

// Store holds per-account window series and the merged hourly-burn shape
// used by the shape-aware model. All fields are guarded by mu; Apply is the
// only writer (sampler goroutines) and Snapshot views are the readers.
// Persistence rides on dirty: Apply appends encoded records, the engine
// drains them on its flush cadence; Restore seeds from a prior file.
type Store struct {
	mu      sync.Mutex
	history time.Duration
	windows map[WindowID][]WindowSample // ascending by Ts, deduped by Ts
	hours   map[string]*hourShape       // keyID → merged hourly usage
	lastOK  map[string]time.Time        // keyID → last successful observation
	labels  map[string]string           // keyID → label
	dirty   []persistRecord
}

// hourShape merges upstream usage buckets into per-epoch-hour totals. A
// full (non-partial) later value always wins; a partial value only fills
// gaps — the trailing partial bucket must never overwrite a complete hour.
type hourShape struct {
	vals map[int64]hourValue
}

type hourValue struct {
	usd  float64
	full bool
}

// windowKind reports the unit semantics of a window name.
func windowKind(name string) string {
	switch name {
	case WindowSession, WindowWeekly:
		return UnitPct
	default:
		return UnitUSD
	}
}

// NewStore returns an empty store retaining history-durable samples.
func NewStore(history time.Duration) *Store {
	if history <= 0 {
		history = DefaultHistory
	}
	return &Store{
		history: history,
		windows: make(map[WindowID][]WindowSample),
		hours:   make(map[string]*hourShape),
		lastOK:  make(map[string]time.Time),
		labels:  make(map[string]string),
	}
}

// Apply folds one observation into the series. Idempotent per Ts: a
// re-observation of the same instant is a no-op, so engine retries never
// duplicate points. The label is tracked for views; hours merge by rule of
// hourShape.
func (s *Store) Apply(snap AccountSnapshot) []persistRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	if snap.At.IsZero() {
		return nil
	}
	at := snap.At.UTC()
	if prev, ok := s.lastOK[snap.KeyID]; ok && !at.After(prev) {
		return nil
	}
	s.lastOK[snap.KeyID] = at
	if snap.Label != "" {
		if s.labels[snap.KeyID] != snap.Label {
			s.labels[snap.KeyID] = snap.Label
			s.dirty = append(s.dirty, labelRecord{KeyID: snap.KeyID, Label: snap.Label})
		}
	}
	addWindows := func(name string, w WindowSample) {
		if w.Ts.IsZero() {
			return
		}
		w.Ts = w.Ts.UTC()
		id := WindowID{KeyID: snap.KeyID, Name: name}
		series := s.windows[id]
		if n := len(series); n > 0 && series[n-1].Ts.Equal(w.Ts) {
			return // same-instant retry
		}
		series = append(series, w)
		series = s.trim(series, w.Ts)
		s.windows[id] = series
		s.dirty = append(s.dirty, sampleRecord{KeyID: snap.KeyID, W: name, S: w})
	}
	if snap.Included != nil && snap.Included.Cap > 0 {
		addWindows(WindowIncluded, WindowSample{
			Ts: at, Remaining: snap.Included.Remaining, Cap: snap.Included.Cap, ResetsAt: snap.Included.ResetsAt,
		})
	}
	if snap.Session != nil {
		addWindows(WindowSession, WindowSample{Ts: at, Remaining: snap.Session.RemainingPercent, ResetsAt: snap.Session.ResetsAt})
	}
	if snap.Weekly != nil {
		addWindows(WindowWeekly, WindowSample{Ts: at, Remaining: snap.Weekly.RemainingPercent, ResetsAt: snap.Weekly.ResetsAt})
	}
	if snap.PurchasedUSD >= 0 {
		addWindows(WindowPurchased, WindowSample{Ts: at, Remaining: snap.PurchasedUSD})
	}
	// Hour buckets: merge by completeness rule.
	for _, h := range snap.Hours {
		if h.HourStart.IsZero() {
			continue
		}
		rec := hourRecord{
			KeyID: snap.KeyID, EpochHour: HourStart(h.HourStart), USD: h.UsageUSD, Full: !h.Partial,
		}
		if s.mergeHour(rec) {
			s.dirty = append(s.dirty, rec)
		}
	}
	return nil
}

// mergeHour folds one bucket per the completeness rule; true only when the
// stored value actually changed (new bucket, completeness upgrade, or a
// revised number) so the record must persist. D10 write-amplification fix:
// the 24h usage range re-reports every bucket as FULL for a day after it
// completes — an unguarded overwrite re-marked them all every sample
// (600k of 679k persisted lines were re-reports of unchanged hours).
func (s *Store) mergeHour(rec hourRecord) bool {
	shape := s.hours[rec.KeyID]
	if shape == nil {
		shape = &hourShape{vals: make(map[int64]hourValue)}
		s.hours[rec.KeyID] = shape
	}
	prev, ok := shape.vals[rec.EpochHour]
	if ok && !rec.Full {
		return false // a complete entry beats any later partial
	}
	if ok && prev.full == rec.Full && prev.usd == rec.USD {
		return false // identical re-report: nothing changed
	}
	shape.vals[rec.EpochHour] = hourValue{usd: rec.USD, full: rec.Full}
	s.trimHours(shape)
	return true
}

// trimHours drops hour buckets older than the retention window
// (opportunistic, per-merge).
func (s *Store) trimHours(shape *hourShape) {
	cutoff := HourStart(time.Now().UTC()) - int64(s.history/time.Hour) - 1
	for hs := range shape.vals {
		if hs < cutoff {
			delete(shape.vals, hs)
		}
	}
}

// trim drops samples older than retention; cutoff derives from the newest
// timestamp so restored-but-old series age out regardless of wall clock.
func (s *Store) trim(series []WindowSample, newest time.Time) []WindowSample {
	cutoff := newest.Add(-s.history)
	keep := series[:0]
	for _, sm := range series {
		if !sm.Ts.Before(cutoff) {
			keep = append(keep, sm)
		}
	}
	return keep
}

// Series returns the retained samples of one window (ascending).
func (s *Store) Series(keyID, window string) []WindowSample {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]WindowSample, 0, len(s.windows[WindowID{KeyID: keyID, Name: window}]))
	return append(out, s.windows[WindowID{KeyID: keyID, Name: window}]...)
}

// LastOK reports when a key last yielded a successful observation.
func (s *Store) LastOK(keyID string) (time.Time, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.lastOK[keyID]
	return t, ok
}

// Label returns the stored label for a key ("" when unknown).
func (s *Store) Label(keyID string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.labels[keyID]
}

// Keys lists keyIDs known to the store (stable order not guaranteed).
func (s *Store) Keys() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.lastOK))
	for k := range s.lastOK {
		out = append(out, k)
	}
	return out
}

// HourProfile returns the mean burn (USD) per hour-of-day across retained
// hours, and whether any coverage exists. Partial-only hours still count
// once a full value exists (the merge rule keeps the best).
func (s *Store) HourProfile(keyID string) (map[int]float64, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	shape := s.hours[keyID]
	if shape == nil || len(shape.vals) == 0 {
		return nil, false
	}
	sums := make(map[int]float64, 24)   // hour-of-day → ΣUSD
	counts := make(map[int]float64, 24) // hour-of-day → n hours observed
	for hs, hv := range shape.vals {
		hod := int(time.Unix(hs*3600, 0).UTC().Hour())
		sums[hod] += hv.usd
		counts[hod]++
	}
	profile := make(map[int]float64, 24)
	for hod, sum := range sums {
		profile[hod] = sum / counts[hod]
	}
	return profile, true
}

// Stale reports whether a key's last observation predates maxAge.
func (s *Store) Stale(keyID string, maxAge time.Duration) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.lastOK[keyID]
	if !ok {
		return true
	}
	return time.Since(t) > maxAge
}

// WindowNames returns every window with data for one key (plan order).
func (s *Store) WindowNames(keyID string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	order := []string{WindowIncluded, WindowSession, WindowWeekly, WindowPurchased}
	out := make([]string, 0, 4)
	for _, name := range order {
		if len(s.windows[WindowID{KeyID: keyID, Name: name}]) > 0 {
			out = append(out, name)
		}
	}
	return out
}

// DrainDirty atomically returns and clears buffered persistence records.
func (s *Store) DrainDirty() []persistRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.dirty) == 0 {
		return nil
	}
	out := s.dirty
	s.dirty = nil
	return out
}

// Restore seeds the store from persisted records and applies the retention
// policy (plan §13 D10): restored series trim against their own newest
// sample (aging regardless of wall clock, matching live Apply), and hour
// buckets cut at the retention floor — restored state obeys the same
// retention the engine uses live, so the post-restore compaction re-emits
// only retained state.
func (s *Store) Restore(recs []persistRecord, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range recs {
		s.ingestLocked(r)
	}
	s.trimRetentionLocked(now)
}

// IngestRecord folds one persisted record without retention work — the
// streaming-load fast path (D10): records flow in line by line and memory
// stays bounded by retention only once TrimRetention closes the pass.
func (s *Store) IngestRecord(rec persistRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ingestLocked(rec)
}

// TrimRetention applies the retention policy after a streaming ingest.
func (s *Store) TrimRetention(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.trimRetentionLocked(now)
}

func (s *Store) ingestLocked(rec persistRecord) {
	switch v := rec.(type) {
	case sampleRecord:
		id := WindowID{KeyID: v.KeyID, Name: v.W}
		series := s.windows[id]
		dup := false
		for _, sm := range series {
			if sm.Ts.Equal(v.S.Ts) {
				dup = true
				break
			}
		}
		if !dup {
			series = append(series, v.S)
		}
		s.windows[id] = series
		if v.S.Ts.After(s.lastOK[v.KeyID]) {
			s.lastOK[v.KeyID] = v.S.Ts
		}
		if _, ok := s.labels[v.KeyID]; !ok {
			s.labels[v.KeyID] = "Ollama credential"
		}
	case hourRecord:
		s.mergeHour(v)
	case labelRecord:
		s.labels[v.KeyID] = v.Label
	}
}

// trimRetentionLocked runs the retention sweep: per-series self-anchored
// TTL for samples, wall-clock floor for hour buckets.
func (s *Store) trimRetentionLocked(_ time.Time) {
	for id, series := range s.windows {
		var newest time.Time
		for _, sm := range series {
			if sm.Ts.After(newest) {
				newest = sm.Ts
			}
		}
		if newest.IsZero() {
			delete(s.windows, id)
			continue
		}
		trimmed := s.trim(series, newest)
		switch {
		case len(trimmed) == 0:
			delete(s.windows, id)
		case len(trimmed) != len(series):
			s.windows[id] = trimmed
		}
	}
	for id, shape := range s.hours {
		s.trimHours(shape)
		if len(shape.vals) == 0 {
			delete(s.hours, id)
		}
	}
}

// SnapshotRecords encodes the entire retained state as persist records —
// the compaction payload used right after a load (plan §7: the file stops
// growing across restarts). Identifying data is digest-only.
func (s *Store) SnapshotRecords() []persistRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]persistRecord, 0, 4096)
	for k, l := range s.labels {
		out = append(out, labelRecord{KeyID: k, Label: l})
	}
	for id, series := range s.windows {
		for _, sm := range series {
			out = append(out, sampleRecord{KeyID: id.KeyID, W: id.Name, S: sm})
		}
	}
	for k, shape := range s.hours {
		for hs, hv := range shape.vals {
			out = append(out, hourRecord{KeyID: k, EpochHour: hs, USD: hv.usd, Full: hv.full})
		}
	}
	return out
}

// HourStart truncates a time to its hour-aligned epoch hour.
func HourStart(t time.Time) int64 {
	u := t.UTC()
	return time.Date(u.Year(), u.Month(), u.Day(), u.Hour(), 0, 0, 0, time.UTC).Unix() / 3600
}

// DefaultHistory mirrors the config default for zero-arg stores in tests.
const DefaultHistory = 7 * 24 * time.Hour
