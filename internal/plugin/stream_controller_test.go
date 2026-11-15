package plugin

import (
	"sync/atomic"
	"testing"
	"time"

	"ollama-cliproxyapi/internal/config"
)

func TestStreamLimitsZeroConfigUsesDefaults(t *testing.T) {
	l := limitsFromConfig(config.Config{})
	if l.first != config.DefaultStreamFirstDataTimeout || l.idle != config.DefaultStreamIdleTimeout || l.total != config.DefaultStreamTotalTimeout || l.grace != config.DefaultStreamFinishGrace {
		t.Fatalf("limits=%+v", l)
	}
}

func TestStreamControllerStaleGenerationCannotFireAfterReset(t *testing.T) {
	var closes atomic.Int32
	bridge := NewHostBridge(func(string, []byte) ([]byte, error) {
		closes.Add(1)
		return hostOK(map[string]any{}), nil
	})
	c := newStreamController(bridge, "up", time.Now(), streamLimits{total: time.Second, idle: 120 * time.Millisecond})
	c.armPhase(35*time.Millisecond, "first_data")
	time.Sleep(20 * time.Millisecond)
	c.progress()
	time.Sleep(45 * time.Millisecond) // past the stale first-data deadline
	if cause := c.Cause(); cause != "" {
		c.stop()
		t.Fatalf("stale generation fired after reset: %q", cause)
	}
	if got := closes.Load(); got != 0 {
		c.stop()
		t.Fatalf("stale generation closed stream %d times", got)
	}
	c.stop()
}

func TestStreamControllerStopMakesCallbacksHarmless(t *testing.T) {
	var closes atomic.Int32
	bridge := NewHostBridge(func(string, []byte) ([]byte, error) {
		closes.Add(1)
		return hostOK(map[string]any{}), nil
	})
	c := newStreamController(bridge, "up", time.Now(), streamLimits{total: time.Second})
	c.armPhase(30*time.Millisecond, "first_data")
	c.stop()
	time.Sleep(50 * time.Millisecond)
	if cause := c.Cause(); cause != "" {
		t.Fatalf("stopped controller acquired cause %q", cause)
	}
	if got := closes.Load(); got != 0 {
		t.Fatalf("stopped controller closed stream %d times", got)
	}
}

// TestStreamControllerSurvivesNilBridge pins the panic guard on the watchdog
// goroutine: fire runs on a time.AfterFunc goroutine inside the host process,
// and a nil bridge there would otherwise dereference inside the bridge and
// abort the whole CPA process instead of just missing one upstream close.
func TestStreamControllerSurvivesNilBridge(t *testing.T) {
	c := newStreamController(nil, "up", time.Now(), streamLimits{total: time.Second})
	c.armPhase(10*time.Millisecond, "first_data")
	deadline := time.Now().Add(2 * time.Second)
	for c.Cause() == "" && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	c.stop()
	if cause := c.Cause(); cause != "first_data" {
		t.Fatalf("cause = %q, want first_data claimed before the panicking close", cause)
	}
}
