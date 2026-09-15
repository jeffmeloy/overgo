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

// peakHandle would retain a child's memory counters across its exit.
type peakHandle struct{}

func retainPeak(*os.Process) (peakHandle, error) {
	return peakHandle{}, errors.New("process peak measurement is unavailable on this platform")
}

func (peakHandle) read() (uint64, error) {
	return 0, errors.New("process peak measurement is unavailable on this platform")
}

func (peakHandle) close() error { return nil }
