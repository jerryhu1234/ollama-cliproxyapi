package shared

// StreamSnapshot is a read-only, request-local view of converter progress.
// Progress advances only after a complete protocol frame was parsed and made a
// meaningful state transition. Converters are sequential-use; callers must not
// call Feed and StreamSnapshot concurrently.
type StreamSnapshot struct {
	Progress     uint64
	Started      bool
	FinishSeen   bool
	DoneSeen     bool
	TerminalSeen bool
	TerminalKind string
}

// StreamState is embedded by protocol converters to maintain StreamSnapshot.
type StreamState struct{ Snapshot StreamSnapshot }

func (s *StreamState) Start()   { s.Snapshot.Started = true; s.Snapshot.Progress++ }
func (s *StreamState) Advance() { s.Snapshot.Progress++ }
func (s *StreamState) Finish()  { s.Snapshot.FinishSeen = true; s.Snapshot.Progress++ }
func (s *StreamState) Done()    { s.Snapshot.DoneSeen = true; s.Snapshot.Progress++ }
func (s *StreamState) Terminal(kind string) {
	s.Snapshot.TerminalSeen = true
	s.Snapshot.TerminalKind = kind
	s.Snapshot.Progress++
}
