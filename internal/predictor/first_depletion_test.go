package predictor

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

func TestFirstDepletionIdentity(t *testing.T) {
	now := time.Now().UTC()
	for _, mode := range []string{"earliest", "tie", "reset", "stale", "empty"} {
		t.Run(mode, func(t *testing.T) {
			s := NewStore(24 * time.Hour)
			want := "b"
			if mode != "empty" {
				seedLinear(s, "a", now, 4, 55, 50, 100)
				seedLinear(s, "b", now, 4, 34, 30, 100)
			}
			if mode == "tie" {
				s = NewStore(24 * time.Hour)
				seedLinear(s, "a", now, 4, 34, 30, 100)
				seedLinear(s, "b", now, 4, 34, 30, 100)
				want = "a"
			}
			if mode == "reset" {
				reset := now.Add(time.Hour)
				s = NewStore(24 * time.Hour)
				seedLinear(s, "b", now, 4, 34, 30, 100)
				for i, rem := range []float64{30, 10} {
					s.Apply(AccountSnapshot{KeyID: "a", At: now.Add(time.Duration(i-1) * 4 * time.Hour), Included: &USDWindow{Remaining: rem, Cap: 100, ResetsAt: reset}})
				}
			}
			if mode == "stale" {
				s = NewStore(24 * time.Hour)
				seedLinear(s, "a", now, 4, 55, 50, 100)
				seedLinear(s, "b", now.Add(-10*time.Hour), 4, 34, 30, 100)
				want = "a"
			}
			if mode == "empty" {
				want = ""
			}
			view := BuildFleetView(s, []string{"a", "b"}, FleetOptions{Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{}}, nil)
			b, err := json.Marshal(view)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]any
			if err := json.Unmarshal(b, &fields); err != nil {
				t.Fatal(err)
			}
			if want == "" {
				if fields["first_depletion_key_id"] != nil || fields["first_depletion_window"] != nil {
					t.Fatalf("unexpected identity: %s", b)
				}
			} else if fields["first_depletion_key_id"] != want || fields["first_depletion_window"] != WindowIncluded {
				t.Fatalf("expected winner %s/%s: %s", want, WindowIncluded, b)
			}
		})
	}
}

func TestFirstDepletionSkipsFrozenAccounts(t *testing.T) {
	now := time.Now().UTC()
	for _, remaining := range []float64{0, 0.2} {
		t.Run(fmt.Sprintf("weekly_remaining_%g", remaining), func(t *testing.T) {
			s := NewStore(24 * time.Hour)
			// A frozen weekly gate must disqualify the entire account, not
			// just its empty window. A small positive reading is also empty
			// under the percentage epsilon used by the account status.
			for i := 0; i < 3; i++ {
				s.Apply(AccountSnapshot{KeyID: "frozen", At: now.Add(time.Duration(i-2) * time.Hour),
					Session: &PctWindow{RemainingPercent: 95 - 11*float64(i), ResetsAt: now.Add(24 * time.Hour)},
					Weekly:  &PctWindow{RemainingPercent: remaining + 2.1*float64(2-i), ResetsAt: now.Add(72 * time.Hour)}})
			}
			account := buildAccountView(s, "frozen", now, 3*time.Hour, SamplingRateSource{}, nil)
			if account.Status != StatusExhausted {
				t.Fatalf("fixture must be frozen, got %q", account.Status)
			}
			opts := FleetOptions{Now: now, Horizon: 48 * time.Hour, StaleAfter: 3 * time.Hour, Src: SamplingRateSource{}}
			seedLinear(s, "ready", now, 4, 34, 30, 100)
			view := BuildFleetView(s, []string{"frozen", "ready"}, opts, nil)
			if view.FirstDepletionKeyID != "ready" || view.FirstDepletionWindow != WindowIncluded || absDiff(view.FirstDepletion, 30*3600) > 1 {
				t.Fatalf("must choose ready account at 30h, got %s/%s at %gs", view.FirstDepletionKeyID, view.FirstDepletionWindow, view.FirstDepletion)
			}
			if view.ExhaustedAccounts != 1 {
				t.Fatalf("frozen account must remain in waiting-reset tally: %d", view.ExhaustedAccounts)
			}
			view = BuildFleetView(s, []string{"frozen"}, opts, nil)
			if view.FirstDepletion != 0 || view.FirstDepletionKeyID != "" || view.FirstDepletionWindow != "" {
				t.Fatalf("all frozen must have no prediction, got %s/%s at %gs", view.FirstDepletionKeyID, view.FirstDepletionWindow, view.FirstDepletion)
			}
			// A newly sampled refill makes this account eligible again;
			// the previous winner is not retained between refreshes.
			later := now.Add(time.Minute)
			s.Apply(AccountSnapshot{KeyID: "frozen", At: later,
				Session: &PctWindow{RemainingPercent: 72, ResetsAt: now.Add(24 * time.Hour)},
				Weekly:  &PctWindow{RemainingPercent: 100, ResetsAt: now.Add(7 * 24 * time.Hour)}})
			opts.Now = later
			view = BuildFleetView(s, []string{"frozen", "ready"}, opts, nil)
			if view.FirstDepletionKeyID != "frozen" || view.ExhaustedAccounts != 0 {
				t.Fatalf("refilled account must re-enter candidates: winner=%q exhausted=%d", view.FirstDepletionKeyID, view.ExhaustedAccounts)
			}
		})
	}
}
