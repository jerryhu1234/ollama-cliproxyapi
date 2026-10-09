//go:build debug

package plugin

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

var debugMu sync.Mutex

// debugTrace is temporary diagnosis instrumentation. It records only control
// flow and non-secret metadata; credentials, headers, prompts, and bodies are
// deliberately excluded.
func debugTrace(format string, args ...any) {
	debugMu.Lock()
	defer debugMu.Unlock()

	path := filepath.Join(filepath.Dir(os.Args[0]), "ollama-debug.log")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return
	}
	defer f.Close()
	values := append([]any{time.Now().Format(time.RFC3339Nano)}, args...)
	_, _ = fmt.Fprintf(f, "%s "+format+"\n", values...)
}
