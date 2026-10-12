package predictor

import (
	"reflect"
	"testing"
	"time"
)

func TestStoreApplyAndSeries(t *testing.T) {
	s := NewStore(24 * time.Hour)
	now := time.Now().UTC()
	// PurchasedUSD = -1 means "absent from the balance shape"; 0 is a real
	// tracked zero-balance row (adapter always publishes the field).
	s.Apply(AccountSnapshot{KeyID: "k1", Label: "L1", At: now.Add(-time.Hour),
		Included: &USDWindow{Remaining: 90, Cap: 100}, PurchasedUSD: -1})
	s.Apply(AccountSnapshot{KeyID: "k1", Label: "L1", At: now,
		Included: &USDWindow{Remaining: 88, Cap: 100}, PurchasedUSD: -1})
	series := s.Series("k1", WindowIncluded)
	if len(series) != 2 || series[1].Remaining != 88 || series[1].Cap != 100 {
		t.Fatalf("series = %+v", series)
	}
	if got := s.Label("k1"); got != "L1" {
		t.Fatalf("label = %q", got)
	}
	// Same-instant re-observation (engine retry) must not duplicate.
	s.Apply(AccountSnapshot{KeyID: "k1", Label: "L1", At: now,
		Included: &USDWindow{Remaining: 88, Cap: 100}})
	if got := s.Series("k1", WindowIncluded); len(got) != 2 {
		t.Fatalf("duplicate sample applied: %+v", got)
	}
	// Out-of-order (older) observation is ignored via lastOK monotonicity.
	s.Apply(AccountSnapshot{KeyID: "k1", Label: "L1", At: now.Add(-2 * time.Hour),
		Included: &USDWindow{Remaining: 99, Cap: 100}})
	if got := s.Series("k1", WindowIncluded); len(got) != 2 || got[1].Remaining != 88 {
		t.Fatalf("stale observation applied: %+v", got)
	}
	// purchased USD=0 is a legitimate tracked row (zero-balance countdown).
	s.Apply(AccountSnapshot{KeyID: "k1", Label: "L1", At: now.Add(time.Minute), PurchasedUSD: 0})
	if got := s.Series("k1", WindowPurchased); len(got) != 1 || got[0].Remaining != 0 {
		t.Fatalf("purchased row = %+v", got)
	}
}

func TestStoreHistoryTrim(t *testing.T) {
	s := NewStore(2 * time.Hour)
	now := time.Now().UTC()
	for i := 3; i >= 0; i-- {
		s.Apply(AccountSnapshot{KeyID: "k", At: now.Add(-time.Duration(i) * time.Hour),
			Included: &USDWindow{Remaining: float64(90 + i), Cap: 100}})
	}
	// Newest sample anchors the cutoff: samples within 2h of it survive —
	// the exact-cutoff boundary is inclusive, so 3 of the 4 remain.
	got := s.Series("k", WindowIncluded)
	if len(got) != 3 || got[0].Remaining != 92 {
		t.Fatalf("trim kept %+v, want the 3 newest", got)
	}
}

func TestStoreHoursShape(t *testing.T) {
	s := NewStore(24 * time.Hour)
	now := time.Now().UTC()
	hour := HourStart(now.Add(-3 * time.Hour))
	// A trailing partial bucket first, then the complete hour later: full wins.
	s.Apply(AccountSnapshot{KeyID: "k", At: now.Add(-2 * time.Hour), Hours: []HourUsage{
		{HourStart: time.Unix(hour*3600, 0), UsageUSD: 1.5, Partial: true},
	}})
	s.Apply(AccountSnapshot{KeyID: "k", At: now.Add(-time.Hour), Hours: []HourUsage{
		{HourStart: time.Unix(hour*3600, 0), UsageUSD: 2.0, Partial: false},
	}})
	// Reverse order attempt afterwards: a late partial must not downgrade.
	s.Apply(AccountSnapshot{KeyID: "k", At: now, Hours: []HourUsage{
		{HourStart: time.Unix(hour*3600, 0), UsageUSD: 0.2, Partial: true},
	}})
	profile, ok := s.HourProfile("k")
	if !ok || len(profile) != 1 {
		t.Fatalf("profile = %+v ok=%t", profile, ok)
	}
	hod := int(time.Unix(hour*3600, 0).UTC().Hour())
	if profile[hod] != 2.0 {
		t.Fatalf("profile[%d] = %v, want 2.0 (full beats partial both ways)", hod, profile[hod])
	}
}

func TestStoreStale(t *testing.T) {
	s := NewStore(24 * time.Hour)
	if !s.Stale("ghost", time.Hour) {
		t.Fatal("unknown key must read stale")
	}
	s.Apply(AccountSnapshot{KeyID: "k", At: time.Now().UTC().Add(-3 * time.Hour),
		Included: &USDWindow{Remaining: 5, Cap: 10}})
	if !s.Stale("k", time.Hour) || s.Stale("k", 4*time.Hour) {
		t.Fatal("stale assessment against maxAge is wrong")
	}
}

func TestStoreWindowNamesOrder(t *testing.T) {
	s := NewStore(24 * time.Hour)
	now := time.Now().UTC()
	s.Apply(AccountSnapshot{KeyID: "k", At: now, Weekly: &PctWindow{RemainingPercent: 80},
		PurchasedUSD: 3})
	if got := s.WindowNames("k"); !reflect.DeepEqual(got, []string{WindowWeekly, WindowPurchased}) {
		t.Fatalf("WindowNames = %v", got)
	}
}

func TestStorePersistenceRoundtrip(t *testing.T) {
	s := NewStore(24 * time.Hour)
	now := time.Now().UTC()
	hour := HourStart(now.Add(-2 * time.Hour))
	s.Apply(AccountSnapshot{KeyID: "k", Label: "L1", At: now.Add(-time.Hour),
		Included: &USDWindow{Remaining: 90, Cap: 100},
		Weekly:   &PctWindow{RemainingPercent: 70, ResetsAt: now.Add(72 * time.Hour)},
		Hours:    []HourUsage{{HourStart: time.Unix(hour*3600, 0), UsageUSD: 2.5, Partial: false}}, PurchasedUSD: -1})
	s.Apply(AccountSnapshot{KeyID: "k", Label: "L1", At: now,
		Included:     &USDWindow{Remaining: 84, Cap: 100},
		PurchasedUSD: 12.5})

	s2 := NewStore(24 * time.Hour)
	s2.Restore(DecodeRecords(EncodeRecords(s.DrainDirty())), now)

	if got := s2.Series("k", WindowIncluded); len(got) != 2 || got[1].Remaining != 84 {
		t.Fatalf("restored included = %+v", got)
	}
	if got := s2.Series("k", WindowWeekly); len(got) != 1 || got[0].Remaining != 70 || got[0].ResetsAt.IsZero() {
		t.Fatalf("restored weekly = %+v", got)
	}
	if got := s2.Series("k", WindowPurchased); len(got) != 1 || got[0].Remaining != 12.5 {
		t.Fatalf("restored purchased = %+v", got)
	}
	if got := s2.Label("k"); got != "L1" {
		t.Fatalf("restored label = %q", got)
	}
	if _, ok := s2.HourProfile("k"); !ok {
		t.Fatal("restored hours missing")
	}
	if _, ok := s2.LastOK("k"); !ok {
		t.Fatal("restored lastOK missing")
	}
	// Compaction: re-encoding the restored state must round-trip again.
	s3 := NewStore(24 * time.Hour)
	s3.Restore(DecodeRecords(EncodeRecords(s2.SnapshotRecords())), now)
	if got := s3.Series("k", WindowIncluded); len(got) != 2 || got[1].Remaining != 84 {
		t.Fatalf("compacted series = %+v", got)
	}
	restored, _ := s3.HourProfile("k")
	orig, _ := s2.HourProfile("k")
	hod := int(time.Unix(hour*3600, 0).UTC().Hour())
	if restored[hod] != orig[hod] {
		t.Fatalf("compacted hours differ: %v vs %v", restored, orig)
	}
}

func TestDecodeRecordsTornTail(t *testing.T) {
	now := time.Now().UTC()
	good := sampleRecord{KeyID: "k", W: WindowIncluded, S: WindowSample{Ts: now, Remaining: 5}}.encode()
	torn := append([]byte("\n{\"t\":\"s\",\"k\":"), []byte("bad\"")...)
	data := append(append([]byte{}, good...), torn...)
	recs := DecodeRecords(data)
	if len(recs) != 1 {
		t.Fatalf("recs = %d, want 1 (torn line skipped)", len(recs))
	}
}
