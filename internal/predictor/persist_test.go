package predictor

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

// hourRecs filters the dirty/snapshot record list to hour records.
func hourRecs(recs []persistRecord) []hourRecord {
	out := make([]hourRecord, 0, len(recs))
	for _, r := range recs {
		if h, ok := r.(hourRecord); ok {
			out = append(out, h)
		}
	}
	return out
}

// fullDayHours builds one account's 24h usage-report shape: 23 complete
// buckets plus a trailing partial — exactly what each sample re-reports up
// to 24h after a bucket completes (the D10 write-amplification source).
func fullDayHours(base time.Time, usdOf func(int) float64) []HourUsage {
	hs := make([]HourUsage, 0, 24)
	for i := 23; i >= 0; i-- {
		hs = append(hs, HourUsage{
			HourStart: base.Add(time.Duration(-i) * time.Hour),
			UsageUSD:  usdOf(23 - i),
			Partial:   i == 0,
		})
	}
	return hs
}

// TestHourReReportsNoWriteAmplification pins the mergeHour fix: every
// sample re-reports the whole 24h usage range; unchanged complete buckets
// must not mark the store dirty (in production 600k of 679k persisted
// lines were identical re-reports of unchanged hour buckets).
func TestHourReReportsNoWriteAmplification(t *testing.T) {
	s := NewStore(7 * 24 * time.Hour) // production TTL: all 24 buckets stay in retention
	now := time.Now().UTC()
	base := HourStart(now.Add(-23 * time.Hour))
	t0 := time.Unix(base*3600, 0).UTC()

	const oneHourUSD = 1.0
	apply := func(at time.Time, usdOf func(int) float64, trailingFull bool) []hourRecord {
		hs := make([]HourUsage, 0, 24)
		for i := 23; i >= 0; i-- {
			idx := 23 - i
			hs = append(hs, HourUsage{
				HourStart: time.Unix((base-int64(i))*3600, 0).UTC(),
				UsageUSD:  usdOf(idx),
				Partial:   i == 0 && !trailingFull,
			})
		}
		if trailingFull {
			hs[len(hs)-1].Partial = false
		}
		s.Apply(AccountSnapshot{KeyID: "k", Label: "L", At: at, Hours: hs})
		return hourRecs(s.DrainDirty())
	}

	flat := func(int) float64 { return oneHourUSD }
	revise := func(i int) float64 {
		if i == 5 {
			return 3.5
		}
		return oneHourUSD
	}

	if got := apply(t0, flat, false); len(got) != 24 {
		t.Fatalf("first pass: every bucket must persist once, dirty = %d", len(got))
	}
	if got := apply(t0.Add(5*time.Minute), flat, false); len(got) != 0 {
		t.Fatalf("identical re-reports must not dirty, got %d", len(got))
	}
	if got := apply(t0.Add(10*time.Minute), revise, false); len(got) != 1 {
		t.Fatalf("one revised bucket must dirty exactly one record, got %d", len(got))
	}
	// Trailing bucket completed between samples: the upgrade is a real
	// change (completeness feeds the diurnal profile weighting). Keep the
	// pass-3 revision in place so only the upgrade dirts.
	sustained := func(i int) float64 {
		if i == 5 {
			return 3.5
		}
		return oneHourUSD
	}
	if got := apply(t0.Add(15*time.Minute), sustained, true); len(got) != 1 {
		t.Fatalf("partial→full upgrade must dirty exactly one record, got %d", len(got))
	}
}

// TestRestoreTrimRetention pins D10 retention parity: restored state obeys
// the same TTL the live engine applies, so a post-restore compaction emits
// only retained state — including for keys that never sample again.
func TestRestoreTrimRetention(t *testing.T) {
	now := time.Now().UTC()
	recs := []persistRecord{
		sampleRecord{KeyID: "k", W: WindowIncluded,
			S: WindowSample{Ts: now.Add(-100 * 24 * time.Hour), Remaining: 1, Cap: 1}},
		sampleRecord{KeyID: "k", W: WindowIncluded,
			S: WindowSample{Ts: now.Add(-time.Hour), Remaining: 5, Cap: 10}},
		hourRecord{KeyID: "k", EpochHour: HourStart(now.Add(-200 * time.Hour)), USD: 9, Full: true},
		hourRecord{KeyID: "k", EpochHour: HourStart(now.Add(-time.Hour)), USD: 2, Full: true},
		labelRecord{KeyID: "k", Label: "L"},
	}
	s := NewStore(7 * 24 * time.Hour)
	s.Restore(recs, now)

	si := s.Series("k", WindowIncluded)
	if len(si) != 1 || si[0].Remaining != 5 {
		t.Fatalf("expired sample must age out, restored = %+v", si)
	}
	prof, ok := s.HourProfile("k")
	if !ok {
		t.Fatal("retained hours must survive restore")
	}
	oldHod := time.Unix(HourStart(now.Add(-200*time.Hour))*3600, 0).UTC().Hour()
	if prof[oldHod] != 0 {
		t.Fatalf("hour beyond retention must drop, profile = %v", prof)
	}
	recentHod := time.Unix(HourStart(now.Add(-time.Hour))*3600, 0).UTC().Hour()
	if prof[recentHod] != 2 {
		t.Fatalf("retained hour must survive, profile = %v", prof)
	}
	if n := len(s.SnapshotRecords()); n != 3 { // 1 sample + 1 hour + 1 label
		t.Fatalf("snapshot must emit retained state only, got %d records", n)
	}
}

// TestPersisterLoadStreamsRecords pins the streaming loader: line-by-line
// decode through a sink, torn-tail tolerance, no size cap.
func TestPersisterLoadStreamsRecords(t *testing.T) {
	now := time.Now().UTC()
	recs := make([]persistRecord, 0, 3000)
	for i := 0; i < 1000; i++ {
		id := "k" + strconv.Itoa(i)
		recs = append(recs,
			sampleRecord{KeyID: id, W: WindowIncluded, S: WindowSample{Ts: now, Remaining: float64(i), Cap: 100}},
			hourRecord{KeyID: id, EpochHour: int64(i) + 1, USD: 1.5, Full: true},
			labelRecord{KeyID: id, Label: "L" + id},
		)
	}
	dir := t.TempDir()
	p := &persister{path: filepath.Join(dir, "usage-samples.ndjson")}
	if err := os.WriteFile(p.path, EncodeRecords(recs), 0o600); err != nil {
		t.Fatal(err)
	}
	count := 0
	labels := map[string]bool{}
	n, err := p.load(func(r persistRecord) {
		count++
		if l, ok := r.(labelRecord); ok {
			labels[l.KeyID] = true
		}
	})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if n != 3000 || count != 3000 {
		t.Fatalf("streamed %d returned / %d fed, want 3000/3000", n, count)
	}
	if !labels["k7"] || labels["k1000"] {
		t.Fatalf("labels misstreamed: %v", labels)
	}

	// Torn tail after a crash: skipped, the valid prefix still restores.
	f, err := os.OpenFile(p.path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("\n{\"t\":\"s\",\"k\":\"k0\",\"w\":\"included\")") // malformed JSON
	_ = f.Close()
	n, err = p.load(func(persistRecord) {})
	if err != nil {
		t.Fatalf("torn tail must not error: %v", err)
	}
	if n != 3000 {
		t.Fatalf("torn tail must be skipped, got %d records", n)
	}
}

// TestPersistStatusRestoreOk pins the healthy path: restore succeeds, the
// status stays ok and the page payload omits the persistence field.
func TestPersistStatusRestoreOk(t *testing.T) {
	dir := t.TempDir()
	now := time.Now().UTC()
	seed := EncodeRecords([]persistRecord{
		sampleRecord{KeyID: "k", W: WindowIncluded, S: WindowSample{Ts: now, Remaining: 5, Cap: 10}},
		labelRecord{KeyID: "k", Label: "L"},
	})
	if err := os.WriteFile(filepath.Join(dir, "usage-samples.ndjson"), seed, 0o600); err != nil {
		t.Fatal(err)
	}
	e := NewEngine(Options{Interval: time.Hour, DataDir: dir})
	e.Start()
	defer e.Stop()
	if st := e.PersistStatus(); st.State != PersistOK {
		t.Fatalf("healthy restore reports %q, want ok", st.State)
	}
	if snap := e.Snapshot(nil); snap.Persistence != nil {
		t.Fatalf("ok persistence must be omitted from the payload, got %+v", snap.Persistence)
	}
	if si := e.Store().Series("k", WindowIncluded); len(si) != 1 {
		t.Fatalf("history not restored: %+v", si)
	}
}

// TestPersistStatusLoadFailureVisible pins D10 visibility: an unloadable
// history file degrades the run with status "load-failed", the page
// payload carries it, and saving is stopped (no rewrite of a file we
// failed to read).
func TestPersistStatusLoadFailureVisible(t *testing.T) {
	dir := t.TempDir()
	hist := filepath.Join(dir, "usage-samples.ndjson")
	if err := os.Mkdir(hist, 0o700); err != nil {
		t.Fatal(err)
	}
	fetch := func(ctx context.Context, key string) (AccountSnapshot, error) {
		return AccountSnapshot{}, nil
	}
	e := NewEngine(Options{Interval: time.Hour, DataDir: dir, Fetch: fetch})
	e.Start()
	defer e.Stop()
	st := e.PersistStatus()
	if st.State != PersistStateLost || st.Detail == "" {
		t.Fatalf("status = %+v, want load-failed with detail", st)
	}
	snap := e.Snapshot(nil)
	if snap.Persistence == nil || snap.Persistence.State != PersistStateLost {
		t.Fatalf("page payload must surface load failure, got %+v", snap.Persistence)
	}
}

// TestFlushCompactsOversizedHistory pins steady-state compaction: a flush
// that observes the file over the threshold rewrites it from retained
// state, so growth self-limits without a restart.
func TestFlushCompactsOversizedHistory(t *testing.T) {
	prev := persistCompactBytes
	persistCompactBytes = 1024
	defer func() { persistCompactBytes = prev }()

	dir := t.TempDir()
	e := NewEngine(Options{Interval: time.Hour, DataDir: dir})
	defer e.Stop()
	e.Start()
	path := e.perse.path

	// Bloated history (junk lines are inert for appends but bulk up size).
	if err := os.WriteFile(path, bytes.Repeat([]byte("x\n"), 2000), 0o600); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	e.Store().Apply(AccountSnapshot{KeyID: "k", Label: "L", At: now,
		Included: &USDWindow{Remaining: 5, Cap: 10}})
	e.flush()

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() >= int64(persistCompactBytes) {
		t.Fatalf("oversized file must compact at flush, size = %d", info.Size())
	}
	// The compacted file must still round-trip.
	compacted := NewStore(7 * 24 * time.Hour)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	compacted.Restore(DecodeRecords(raw), now)
	if got := compacted.Series("k", WindowIncluded); len(got) != 1 || got[0].Remaining != 5 {
		t.Fatalf("compacted file lost the sample: %+v", got)
	}
}

// TestCompactionRewritesCleanOnFailure pins the stray-.tmp cleanup: a
// failed rewrite leaves no half-written temp file behind.
func TestCompactionRewritesCleanOnFailure(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("chmod-based failure test needs a non-root runner")
	}
	dir := t.TempDir()
	p := &persister{path: filepath.Join(dir, "usage-samples.ndjson")}
	if err := os.WriteFile(p.path, EncodeRecords([]persistRecord{
		labelRecord{KeyID: "k", Label: "L"},
	}), 0o600); err != nil {
		t.Fatal(err)
	}
	parent := filepath.Dir(p.path)
	prevMode := mustStatMode(t, parent)
	if err := os.Chmod(parent, 0o500); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Chmod(parent, prevMode); err != nil {
			t.Fatalf("restore dir mode: %v", err)
		}
	}()
	err := p.rewrite([]persistRecord{labelRecord{KeyID: "k2", Label: "X"}})
	if err == nil {
		t.Fatal("rewrite into an unwritable dir must fail")
	}
	if _, serr := os.Stat(p.path + ".tmp"); !os.IsNotExist(serr) {
		t.Fatalf("failed rewrite must not leave a temp file behind")
	}
	// The original file is untouched.
	raw, rerr := os.ReadFile(p.path)
	if rerr != nil || !bytes.Contains(raw, []byte(`"l":"L"`)) {
		t.Fatalf("original file must survive a failed rewrite: %q %v", raw, rerr)
	}
}

func mustStatMode(t *testing.T, path string) os.FileMode {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Mode()
}
