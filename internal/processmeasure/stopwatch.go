package processmeasure

import (
	"errors"
	"time"
)

// Stopwatch is a counter reading taken by NewStopwatch.
type Stopwatch struct {
	started time.Duration
	set     bool
	err     error
}

// NewStopwatch reads the performance counter; a failed read surfaces from Elapsed.
func NewStopwatch() Stopwatch {
	started, err := Counter()
	return Stopwatch{started: started, set: true, err: err}
}

// Elapsed reports the nanoseconds since NewStopwatch, both ends read through the
// counter.
func (s Stopwatch) Elapsed() (uint64, error) {
	switch {
	case !s.set:
		return 0, errors.New("measurement: stopwatch was not started")
	case s.err != nil:
		return 0, s.err
	}
	now, err := Counter()
	if err != nil {
		return 0, err
	}
	if now < s.started {
		return 0, errors.New("measurement: performance counter moved backwards")
	}
	return uint64(now - s.started), nil
}

// Walls reads the stopwatches of one record. Check Err before publication.
type Walls struct {
	Err error // First failed read; successful reads do not clear it.
}

// Elapsed reads a stopwatch; a failed read reports zero and is kept for Err.
func (w *Walls) Elapsed(s Stopwatch) uint64 {
	elapsed, err := s.Elapsed()
	if err != nil && w.Err == nil {
		w.Err = err
	}
	return elapsed
}
