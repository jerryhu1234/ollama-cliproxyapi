package plugin

import (
	"ollama-cliproxyapi/internal/config"
	"sync/atomic"
	"testing"
	"time"
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
