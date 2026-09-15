// Package processlock owns OS file locks released on close or process exit.
package processlock

import (
	"errors"
	"io/fs"
)

// ErrBusy reports contention proven by the OS lock operation. A lock file's
// existence, process name or stale heartbeat does not establish ownership.
var ErrBusy = errors.New("process lock: another open handle holds the lock")

// contendedCause reports a caller that left before acquiring, with the
// contention another handle held as it left: a non-blocking attempt learns
// it, and a lock the attempt won is released at once.
func contendedCause(cause error, path string, mode fs.FileMode) error {
	lock, err := Acquire(path, mode)
	if errors.Is(err, ErrBusy) {
		return errors.Join(cause, ErrBusy)
	}
	_ = lock.Close()
	return cause
}
