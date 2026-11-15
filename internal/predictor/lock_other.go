//go:build !unix

package predictor

import (
	"os"
	"sync"
)

// Non-unix ports (Windows in particular) have no flock in the standard
// library's syscall package and this module carries no extra dependencies, so
// cross-process exclusion degrades to in-process exclusion: a path-keyed
// mutex held for the same one-critical-section lifetime. That is the case
// that matters for the overlap this guards (a reconfigure retiring one engine
// while the new one starts flushes inside the same plugin process). The
// unique temp name in rewrite keeps a rewrite crash-safe regardless.
var (
	lockPathMu sync.Mutex
	lockPaths  = map[string]*sync.Mutex{}
)

func lockFileDesc(_ *os.File, lockPath string) (func(), error) {
	lockPathMu.Lock()
	m, ok := lockPaths[lockPath]
	if !ok {
		m = &sync.Mutex{}
		lockPaths[lockPath] = m
	}
	lockPathMu.Unlock()
	m.Lock()
	return m.Unlock, nil
}
