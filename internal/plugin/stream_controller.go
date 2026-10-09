package plugin

import (
	"sync"
	"time"

	"ollama-cliproxyapi/internal/config"
)

type streamLimits struct{ first, idle, total, grace time.Duration }

func limitsFromConfig(c config.Config) streamLimits {
	l := streamLimits{c.StreamFirstDataTimeout, c.StreamIdleTimeout, c.StreamTotalTimeout, c.StreamFinishGrace}
	if l.first <= 0 {
		l.first = config.DefaultStreamFirstDataTimeout
	}
	if l.idle <= 0 {
		l.idle = config.DefaultStreamIdleTimeout
	}
	if l.total <= 0 {
		l.total = config.DefaultStreamTotalTimeout
	}
	if l.grace <= 0 {
		l.grace = config.DefaultStreamFinishGrace
	}
	return l
}

// streamController owns request-local timers. Callbacks only claim a fixed
// cause and close the upstream stream to release a blocked StreamRead.
type streamController struct {
	mu                     sync.Mutex
	bridge                 *HostBridge
	upstreamID             string
	limits                 streamLimits
	start                  time.Time
	generation             uint64
	active                 bool
	cause                  string
	phaseTimer, totalTimer *time.Timer
	wg                     sync.WaitGroup
}

func newStreamController(bridge *HostBridge, upstreamID string, start time.Time, limits streamLimits) *streamController {
	c := &streamController{bridge: bridge, upstreamID: upstreamID, start: start, limits: limits, active: true}
	c.armTotal(time.Until(start.Add(limits.total)))
	return c
}

func (c *streamController) armTotal(d time.Duration) {
	if d < 0 {
		d = 0
	}
	c.wg.Add(1)
	c.totalTimer = time.AfterFunc(d, func() { defer c.wg.Done(); c.fire(0, "total") })
}
func (c *streamController) armPhase(d time.Duration, cause string) {
	if d < 0 {
		d = 0
	}
	c.mu.Lock()
	c.generation++
	g := c.generation
	if c.phaseTimer != nil && c.phaseTimer.Stop() {
		c.wg.Done()
	}
	c.wg.Add(1)
	c.phaseTimer = time.AfterFunc(d, func() { defer c.wg.Done(); c.fire(g, cause) })
	c.mu.Unlock()
}
func (c *streamController) fire(g uint64, cause string) {
	c.mu.Lock()
	if !c.active || (g != 0 && g != c.generation) || c.cause != "" {
		c.mu.Unlock()
		return
	}
	c.cause = cause
	c.active = false
	c.mu.Unlock()
	_ = c.bridge.StreamClose(c.upstreamID)
}
func (c *streamController) firstData(d time.Duration) { c.armPhase(d, "first_data") }
func (c *streamController) progress()                 { c.armPhase(c.limits.idle, "idle") }
func (c *streamController) finish()                   { c.armPhase(c.limits.grace, "finish_grace") }
func (c *streamController) Cause() string             { c.mu.Lock(); defer c.mu.Unlock(); return c.cause }
func (c *streamController) stop() {
	c.mu.Lock()
	c.active = false
	c.generation++
	if c.phaseTimer != nil && c.phaseTimer.Stop() {
		c.wg.Done()
	}
	if c.totalTimer != nil && c.totalTimer.Stop() {
		c.wg.Done()
	}
	c.mu.Unlock()
	c.wg.Wait()
}
