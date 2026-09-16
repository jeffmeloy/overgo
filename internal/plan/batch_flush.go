package plan

import (
	"errors"
	"fmt"

	"math"
	"time"

	"overgo/internal/textcheck"
	"overgo/internal/worklease"
)

// BatchFlush bounds one keyed batch; flush when any bound is met; every bound
// explicit, none defaulted; keys never mix.
type BatchFlush struct {
	Key         string `json:"key"`
	MaxSize     int    `json:"max_size"`
	MaxInterval string `json:"max_interval"`
	MaxBytes    int64  `json:"max_bytes"`
}

// BatchState records one key's accumulation since its last flush.
type BatchState struct {
	Key       string
	Size      int
	Bytes     int64
	Elapsed   time.Duration
	StartedAt time.Time
}

// FlushReason names why a batch flushes; closed vocabulary.
type FlushReason string

const (
	// FlushNone means no bound met.
	FlushNone FlushReason = ""
	// FlushSize means the size bound met.
	FlushSize FlushReason = "size"
	// FlushInterval means the interval bound met.
	FlushInterval FlushReason = "interval"
	// FlushBytes means the byte bound met.
	FlushBytes    FlushReason = "bytes"
	flushBoundary FlushReason = "boundary"
)

// Interval validates the declaration and returns its elapsed-time bound.
func (f BatchFlush) Interval() (time.Duration, error) {
	interval, err := time.ParseDuration(f.MaxInterval)
	if err != nil || min(int64(interval), int64(f.MaxSize), f.MaxBytes) <= 0 || !textcheck.LowerIdentifier(f.Key, worklease.AutomationTextMaxBytes) {
		return 0, fmt.Errorf("plan: batch flush requires a valid key and positive bounds: %+v", f)
	}
	return interval, nil
}

// Advance applies one optional member to a persisted state and evaluates flush
// bounds. Nil bytes checks elapsed time only. The caller clears the state only
// after accepting the flush, so failed publication cannot discard pending work.
func (f BatchFlush) Advance(state BatchState, bytes *int64, now time.Time, boundary bool) (BatchState, bool, FlushReason, error) {
	interval, err := f.Interval()
	if err != nil {
		return BatchState{}, false, FlushNone, err
	}
	if state == (BatchState{}) {
		state.Key = f.Key
	}
	empty := state.Size == 0
	if state.Key != f.Key || min(int64(state.Size), state.Bytes, int64(state.Elapsed)) < 0 || now.IsZero() ||
		empty && state != (BatchState{Key: f.Key}) ||
		!empty && (state.StartedAt.IsZero() || now.Before(state.StartedAt)) {
		return BatchState{}, false, FlushNone, errors.New("plan: invalid batch state or time")
	}
	if bytes != nil {
		if *bytes < 0 || state.Size == math.MaxInt || *bytes > math.MaxInt64-state.Bytes {
			return BatchState{}, false, FlushNone, errors.New("plan: invalid or overflowing batch member")
		}
		if empty {
			state.StartedAt = now
		}
		state.Size++
		state.Bytes += *bytes
	} else if empty {
		return state, false, FlushNone, nil
	}
	state.Elapsed = now.Sub(state.StartedAt)
	reason := FlushNone
	switch {
	case state.Size >= f.MaxSize:
		reason = FlushSize
	case state.Elapsed >= interval:
		reason = FlushInterval
	case state.Bytes >= f.MaxBytes:
		reason = FlushBytes
	case boundary:
		reason = flushBoundary
	}
	return state, reason != FlushNone, reason, nil
}
