package predictor

import (
	"bytes"
	"encoding/json"
	"time"
)

// persistRecord is one NDJSON line kind. Records are small, append-only,
// and keyed by digest identity; the engine owns the file, the store only
// produces/consumes them. Records carry only digest identities and numeric
// series — never credentials (plan §10 data-safety boundary).
type persistRecord interface {
	encode() []byte
}

// sampleWire is one window-series point.
type sampleWire struct {
	T  string  `json:"t"` // fixed "s"
	K  string  `json:"k"` // keyID digest
	W  string  `json:"w"` // window name
	TS int64   `json:"ts"`
	R  float64 `json:"r"`
	C  float64 `json:"c,omitempty"`
	RA int64   `json:"ra,omitempty"`
}

type sampleRecord struct {
	KeyID string
	W     string // window name
	S     WindowSample
}

func (r sampleRecord) encode() []byte {
	w := sampleWire{T: "s", K: r.KeyID, W: r.W, TS: r.S.Ts.Unix(), R: r.S.Remaining, C: r.S.Cap}
	if !r.S.ResetsAt.IsZero() {
		w.RA = r.S.ResetsAt.Unix()
	}
	raw, _ := json.Marshal(w)
	return raw
}

// hourWire is one merged usage hour.
type hourWire struct {
	T string  `json:"t"` // fixed "h"
	K string  `json:"k"`
	H int64   `json:"h"` // hour-aligned epoch hour
	U float64 `json:"u"`
	F bool    `json:"f,omitempty"` // complete hour
}

type hourRecord struct {
	KeyID     string
	EpochHour int64
	USD       float64
	Full      bool
}

func (r hourRecord) encode() []byte {
	raw, _ := json.Marshal(hourWire{T: "h", K: r.KeyID, H: r.EpochHour, U: r.USD, F: r.Full})
	return raw
}

// labelWire is the display label of a key.
type labelWire struct {
	T string `json:"t"` // fixed "l"
	K string `json:"k"`
	L string `json:"l"`
}

type labelRecord struct {
	KeyID string
	Label string
}

func (r labelRecord) encode() []byte {
	raw, _ := json.Marshal(labelWire{T: "l", K: r.KeyID, L: r.Label})
	return raw
}

// DecodeRecord parses one NDJSON line into a record; ok=false skips it
// (a torn final line after a crash must not break startup). The streaming
// loader (D10) drives this per line.
func DecodeRecord(line []byte) (persistRecord, bool) {
	var head sampleWire // shape-compatible probe for T/K
	if json.Unmarshal(line, &head) != nil || head.K == "" {
		return nil, false
	}
	switch head.T {
	case "s":
		if head.W == "" || head.TS == 0 {
			return nil, false
		}
		rec := sampleRecord{
			KeyID: head.K, W: head.W,
			S: WindowSample{Ts: time.Unix(head.TS, 0).UTC(), Remaining: head.R, Cap: head.C},
		}
		if head.RA != 0 {
			rec.S.ResetsAt = time.Unix(head.RA, 0).UTC()
		}
		return rec, true
	case "h":
		var w hourWire
		if json.Unmarshal(line, &w) != nil || w.H == 0 {
			return nil, false
		}
		return hourRecord{KeyID: head.K, EpochHour: w.H, USD: w.U, Full: w.F}, true
	case "l":
		var w labelWire
		if json.Unmarshal(line, &w) != nil || w.L == "" {
			return nil, false
		}
		return labelRecord{KeyID: head.K, Label: w.L}, true
	}
	return nil, false
}

// DecodeRecords parses NDJSON bytes into records; malformed lines are
// skipped (a torn final line after a crash must not break startup).
// Retained for tests and offline tooling — the engine loads via streaming.
func DecodeRecords(data []byte) []persistRecord {
	var out []persistRecord
	for len(data) > 0 {
		var line []byte
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			line, data = data[:i], data[i+1:]
		} else {
			line, data = data, nil
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		if rec, ok := DecodeRecord(line); ok {
			out = append(out, rec)
		}
	}
	return out
}

// EncodeRecords joins records for a full-file rewrite.
func EncodeRecords(recs []persistRecord) []byte {
	var buf bytes.Buffer
	for _, r := range recs {
		buf.Write(r.encode())
		buf.WriteByte('\n')
	}
	return buf.Bytes()
}

// FormatTimestamp is the RFC3339 view helper ("" for zero times).
func FormatTimestamp(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(time.RFC3339)
}
