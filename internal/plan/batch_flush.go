package plan

import (
	"errors"
	"fmt"
	"time"

	"overgo/internal/textcheck"
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
	Key     string
	Size    int
	Bytes   int64
	Elapsed time.Duration
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
	FlushBytes FlushReason = "bytes"
)

// Interval parses MaxInterval; validated declarations never fail here.
func (f BatchFlush) Interval() (time.Duration, error) {
	interval, err := time.ParseDuration(f.MaxInterval)
	if err != nil || interval <= 0 {
		return 0, errors.New("plan: batch flush interval must be a positive duration")
	}
	return interval, nil
}

// Decide is pure; first bound met in size -> interval -> bytes order names the
// reason; a state for another key never flushes this batch.
func (f BatchFlush) Decide(state BatchState) (bool, FlushReason) {
	if state.Key != f.Key {
		return false, FlushNone
	}
	interval, err := f.Interval()
	if err != nil {
		return false, FlushNone
	}
	switch {
	case state.Size >= f.MaxSize:
		return true, FlushSize
	case state.Elapsed >= interval:
		return true, FlushInterval
	case state.Bytes >= f.MaxBytes:
		return true, FlushBytes
	}
	return false, FlushNone
}

func validateBatchFlush(flush *BatchFlush) error {
	if flush == nil {
		return nil
	}
	if !textcheck.LowerIdentifier(flush.Key, automationRoleMaxBytes) {
		return fmt.Errorf("plan: batch flush key %q is not a lower identifier", flush.Key)
	}
	if flush.MaxSize <= 0 || flush.MaxBytes <= 0 {
		return errors.New("plan: batch flush requires positive size and byte bounds")
	}
	_, err := flush.Interval()
	return err
}

// BatchAccumulator accumulates per key against one declaration; Add records
// one member and returns the decision; a flush resets that key only.
type BatchAccumulator struct {
	flush  BatchFlush
	states map[string]*batchAccumulation
}

type batchAccumulation struct {
	size  int
	bytes int64
	since time.Time
}

// NewBatchAccumulator validates the declaration -> empty accumulator.
func NewBatchAccumulator(flush BatchFlush) (*BatchAccumulator, error) {
	if err := validateBatchFlush(&flush); err != nil {
		return nil, err
	}
	return &BatchAccumulator{flush: flush, states: map[string]*batchAccumulation{}}, nil
}

// Add records one member of key with payload bytes at now -> state after the
// add and the decision; keys other than the declared one are refused.
func (a *BatchAccumulator) Add(key string, payloadBytes int64, now time.Time) (BatchState, bool, FlushReason, error) {
	if a == nil || key != a.flush.Key || payloadBytes < 0 {
		return BatchState{}, false, FlushNone, errors.New("plan: batch accumulator refuses a member outside its declared key")
	}
	current, found := a.states[key]
	if !found {
		current = &batchAccumulation{since: now}
		a.states[key] = current
	}
	current.size++
	current.bytes += payloadBytes
	state := BatchState{Key: key, Size: current.size, Bytes: current.bytes, Elapsed: now.Sub(current.since)}
	flush, reason := a.flush.Decide(state)
	if flush {
		delete(a.states, key)
	}
	return state, flush, reason, nil
}

// Pending counts members accumulated for key and not yet flushed.
func (a *BatchAccumulator) Pending(key string) int {
	if a == nil || a.states[key] == nil {
		return 0
	}
	return a.states[key].size
}
