//go:build !windows

package processmeasure

import (
	"errors"
	"os"
	"time"
)

var counterOrigin = time.Now()

// Counter returns a monotonic timestamp for interval subtraction.
// Its origin is arbitrary; it is not a wall-clock time or a persisted identity.
func Counter() (time.Duration, error) {
	return time.Since(counterOrigin), nil
}

func peakWorkingSet(*os.Process) (uint64, error) {
	return 0, errors.New("process peak measurement is unavailable on this platform")
}
