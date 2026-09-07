// Package processlock owns non-blocking, process-lifetime file locks.
package processlock

import "errors"

// ErrBusy reports contention proven by the OS lock operation. A lock file's
// existence, process name or stale heartbeat does not establish ownership.
var ErrBusy = errors.New("process lock: another open handle holds the lock")
