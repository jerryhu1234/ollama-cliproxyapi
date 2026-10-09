package shared

import "testing"

func TestStreamStateSnapshot(t *testing.T) {
	var s StreamState
	s.Start()
	s.Advance()
	s.Finish()
	s.Done()
	s.Terminal("completed")
	got := s.Snapshot
	if got.Progress != 5 || !got.Started || !got.FinishSeen || !got.DoneSeen || !got.TerminalSeen || got.TerminalKind != "completed" {
		t.Fatalf("snapshot=%+v", got)
	}
}
