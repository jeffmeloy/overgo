//go:build !windows

package processmeasure

import (
	"time"
)

var counterOrigin = time.Now()

// Counter returns a monotonic timestamp for interval subtraction.
// Its origin is arbitrary; it is not a wall-clock time or a persisted identity.
func Counter() (time.Duration, error) {
	return time.Since(counterOrigin), nil
}
