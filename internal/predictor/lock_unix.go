//go:build unix

package predictor

import (
	"os"
	"syscall"
)

// lockFileDesc takes an exclusive advisory flock on f's open file description
// and returns the matching unlock. flock is per open file description, so two
// distinct opens of the same lock file contend in-process as well as across
// processes; the caller already holds the persister mutex for its own handle.
func lockFileDesc(f *os.File, _ string) (func(), error) {
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		return func() {}, err
	}
	return func() {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
	}, nil
}
