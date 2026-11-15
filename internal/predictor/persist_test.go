package predictor

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"sync"
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
// failed rewrite leaves no half-written temp file behind and never
// truncates, renames over, or otherwise disturbs the original history.
func TestCompactionRewritesCleanOnFailure(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("chmod-based failure test needs a non-root runner")
	}
	dir := t.TempDir()
	p := &persister{path: filepath.Join(dir, "usage-samples.ndjson")}
	original := EncodeRecords([]persistRecord{labelRecord{KeyID: "k", Label: "L"}})
	if err := os.WriteFile(p.path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	// Pre-create the flock sidecar so the failure lands on CreateTemp (the
	// rewrite's own write path) rather than on acquiring the lock.
	if err := os.WriteFile(p.path+".lock", nil, 0o600); err != nil {
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
	if residue := tempResidue(t, dir); len(residue) != 0 {
		t.Fatalf("failed rewrite must not leave temp files behind: %v", residue)
	}
	// The original file survives byte for byte — no zero-length or torn path.
	raw, rerr := os.ReadFile(p.path)
	if rerr != nil {
		t.Fatalf("original file must survive a failed rewrite: %v", rerr)
	}
	if !bytes.Equal(raw, original) {
		t.Fatalf("failed rewrite mutated the history: %q, want %q", raw, original)
	}
}

// TestCompactionRenameFailureKeepsOriginal injects the other half of the
// partial-failure contract: the temp write and its fsync succeed but the
// rename target is unusable (p.path is a non-empty directory, which rename(2)
// refuses to replace with a regular file). rewrite must report the failure,
// leave no temp residue, and leave the destination untouched.
func TestCompactionRenameFailureKeepsOriginal(t *testing.T) {
	dir := t.TempDir()
	p := &persister{path: filepath.Join(dir, "usage-samples.ndjson")}
	if err := os.Mkdir(p.path, 0o700); err != nil {
		t.Fatal(err)
	}
	marker := filepath.Join(p.path, "original")
	if err := os.WriteFile(marker, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := p.rewrite([]persistRecord{labelRecord{KeyID: "k2", Label: "X"}}); err == nil {
		t.Fatal("rewrite onto a non-empty directory must fail")
	}
	if raw, rerr := os.ReadFile(marker); rerr != nil || string(raw) != "keep" {
		t.Fatalf("failed rename damaged the destination: %q %v", raw, rerr)
	}
	if residue := tempResidue(t, dir); len(residue) != 0 {
		t.Fatalf("failed rename must not leave temp files behind: %v", residue)
	}
}

// tempResidue lists leftover rewrite temp files (*.tmp) in dir.
func tempResidue(t *testing.T, dir string) []string {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(dir, "*.tmp"))
	if err != nil {
		t.Fatalf("glob temp files: %v", err)
	}
	return matches
}

func mustStatMode(t *testing.T, path string) os.FileMode {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Mode()
}

// TestRewritesSerializeOnSharedPath pins the dual-engine persist contract:
// two persister instances (an old engine's final flush racing a new engine's
// compaction) share one history path. The tmp file must be unique per
// rewrite — a fixed path+".tmp" makes the losing rename fail with ENOENT
// after the winner moved it, and a concurrent truncation corrupts the other
// writer's temp contents.
func TestRewritesSerializeOnSharedPath(t *testing.T) {
	dir := t.TempDir()
	p1 := &persister{path: filepath.Join(dir, "usage-samples.ndjson")}
	p2 := &persister{path: p1.path}
	a := []persistRecord{labelRecord{KeyID: "k", Label: "A"}}
	b := []persistRecord{labelRecord{KeyID: "k", Label: "B"}}
	// Releases both writers through one gate per round so the interleaving
	// (winner rename, loser rename) actually happens instead of the scheduler
	// running them strictly back to back.
	const rounds = 8
	for r := 0; r < rounds; r++ {
		var wg sync.WaitGroup
		start := make(chan struct{})
		errs := make([]error, 2)
		for i, pw := range []struct {
			p    *persister
			recs []persistRecord
		}{{p1, a}, {p2, b}} {
			wg.Add(1)
			go func(i int, p *persister, recs []persistRecord) {
				defer wg.Done()
				<-start
				errs[i] = p.rewrite(recs)
			}(i, pw.p, pw.recs)
		}
		close(start)
		wg.Wait()
		for i, err := range errs {
			if err != nil {
				t.Fatalf("round %d: concurrent rewrite %d failed: %v", r, i, err)
			}
		}
		// The survivor is exactly one writer's full record set: no partial
		// line, no interleaved payload from the loser.
		data, err := os.ReadFile(p1.path)
		if err != nil {
			t.Fatalf("round %d: read history: %v", r, err)
		}
		lines := bytes.Split(bytes.TrimRight(data, "\n"), []byte("\n"))
		if len(lines) != 1 {
			t.Fatalf("round %d: history holds %d lines, want exactly one writer's full set", r, len(lines))
		}
		rec, ok := DecodeRecord(lines[0])
		if !ok {
			t.Fatalf("round %d: torn line after concurrent rewrite: %q", r, lines[0])
		}
		if lbl, ok := rec.(labelRecord); !ok || (lbl.Label != "A" && lbl.Label != "B") {
			t.Fatalf("round %d: history holds a foreign record: %+v", r, rec)
		}
	}
	if residue := tempResidue(t, dir); len(residue) != 0 {
		t.Fatalf("concurrent rewrites left temp residue: %v", residue)
	}
}

// TestAppendAndRewriteSerialize pins the append/rewrite half of the lock: an
// append racing a compaction on one path must never land inside the other's
// payload. Whichever order wins, the file stays a sequence of complete
// records — either the rewrite's snapshot plus the append, or just the
// rewrite's snapshot (the append was replaced wholesale, which is allowed:
// compaction is authoritative for retained state).
func TestAppendAndRewriteSerialize(t *testing.T) {
	dir := t.TempDir()
	p := &persister{path: filepath.Join(dir, "usage-samples.ndjson")}
	snapshot := []persistRecord{labelRecord{KeyID: "k", Label: "snapshot"}}
	appended := []persistRecord{sampleRecord{
		KeyID: "k", W: WindowIncluded,
		S: WindowSample{Ts: time.Now().UTC(), Remaining: 5, Cap: 10},
	}}
	const rounds = 8
	for r := 0; r < rounds; r++ {
		var wg sync.WaitGroup
		start := make(chan struct{})
		var werr, aerr error
		wg.Add(2)
		go func() {
			defer wg.Done()
			<-start
			werr = p.rewrite(snapshot)
		}()
		go func() {
			defer wg.Done()
			<-start
			aerr = p.append(appended)
		}()
		close(start)
		wg.Wait()
		if werr != nil || aerr != nil {
			t.Fatalf("round %d: rewrite=%v append=%v", r, werr, aerr)
		}
		data, err := os.ReadFile(p.path)
		if err != nil {
			t.Fatalf("round %d: read history: %v", r, err)
		}
		for i, line := range bytes.Split(bytes.TrimRight(data, "\n"), []byte("\n")) {
			if len(bytes.TrimSpace(line)) == 0 {
				continue
			}
			if _, ok := DecodeRecord(line); !ok {
				t.Fatalf("round %d: line %d torn after append/rewrite race: %q", r, i, line)
			}
		}
	}
	if residue := tempResidue(t, dir); len(residue) != 0 {
		t.Fatalf("append/rewrite race left temp residue: %v", residue)
	}
}

// TestAppendThenCrashRecoversFlushedWindow pins the durability contract: a
// hard kill may lose at most the one flush window that was in flight, so
// everything an append reported as written must come back after reopen, and a
// torn tail from the kill must be skipped by the loader without an error.
// The kill is simulated by truncating the file mid-record after the
// acknowledged appends — the on-disk state a kill leaves behind.
func TestAppendThenCrashRecoversFlushedWindow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "usage-samples.ndjson")
	now := time.Now().UTC()
	p := &persister{path: path}

	durable := []persistRecord{
		sampleRecord{KeyID: "a", W: WindowIncluded, S: WindowSample{Ts: now, Remaining: 9, Cap: 10}},
		sampleRecord{KeyID: "b", W: WindowIncluded, S: WindowSample{Ts: now, Remaining: 8, Cap: 10}},
	}
	if err := p.append(durable); err != nil {
		t.Fatalf("append: %v", err)
	}
	// A second acknowledged window: two flush ticks survive the kill.
	if err := p.append([]persistRecord{
		sampleRecord{KeyID: "c", W: WindowIncluded, S: WindowSample{Ts: now, Remaining: 7, Cap: 10}},
	}); err != nil {
		t.Fatalf("append: %v", err)
	}

	// Simulate the kill landing mid-write of the next window: a partial record
	// reaches the file but never completes.
	inFlight := EncodeRecords([]persistRecord{
		sampleRecord{KeyID: "d", W: WindowIncluded, S: WindowSample{Ts: now, Remaining: 6, Cap: 10}},
	})
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("open for torn tail: %v", err)
	}
	if _, err := f.Write(inFlight[:len(inFlight)/2]); err != nil {
		t.Fatalf("write torn tail: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close torn tail: %v", err)
	}

	// Reopen (fresh engine) and restore: no fsync bookkeeping in userspace, so
	// this is exactly the post-crash read path.
	reopened := &persister{path: path}
	seen := map[string]bool{}
	n, err := reopened.load(func(rec persistRecord) {
		if s, ok := rec.(sampleRecord); ok {
			seen[s.KeyID] = true
		}
	})
	if err != nil {
		t.Fatalf("post-crash load must not error on a torn tail: %v", err)
	}
	if n != 3 {
		t.Fatalf("restored %d records, want the 3 from acknowledged windows", n)
	}
	for _, id := range []string{"a", "b", "c"} {
		if !seen[id] {
			t.Fatalf("acknowledged sample %q lost across the crash: %v", id, seen)
		}
	}
	if seen["d"] {
		t.Fatalf("the in-flight window must be the only loss, but %q decoded", "d")
	}
}
