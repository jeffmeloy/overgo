package loop

import (
	"errors"
	"time"
)

// backoffShiftLimit bounds the doubling exponent so the shift never
// overflows a Duration before the maximum clamps it.
const backoffShiftLimit = 62

// Backoff is the requeue delay declaration: the delay doubles per attempt
// from Base, a Jitter fraction spreads it, Max clamps it, and the doubling
// is guarded against overflow.
type Backoff struct {
	Base   time.Duration `json:"base"`
	Max    time.Duration `json:"max"`
	Jitter float64       `json:"jitter,omitzero"`
}

// Validate requires a positive base at or below the maximum and a jitter
// fraction within [0, 1]; a zero declaration means no delay.
func (b Backoff) Validate() error {
	if b == (Backoff{}) {
		return nil
	}
	if b.Base <= 0 || b.Max < b.Base {
		return errors.New("loop: backoff needs a positive base at or below its maximum")
	}
	if b.Jitter < 0 || b.Jitter > 1 {
		return errors.New("loop: backoff jitter must lie within [0, 1]")
	}
	return nil
}

// Delay returns the delay before requeue attempt (1-based); unit is a
// number in [0, 1) that places the jitter. Attempt zero and a zero
// declaration delay nothing; the exponent is bounded so doubling cannot
// overflow, and the result never exceeds Max.
func (b Backoff) Delay(attempt int, unit float64) time.Duration {
	if b == (Backoff{}) || attempt <= 0 {
		return 0
	}
	exponent := min(attempt-1, backoffShiftLimit)
	delay := b.Max
	if scaled := b.Base << exponent; exponent < backoffShiftLimit && scaled > 0 && scaled/b.Base == 1<<exponent && scaled < b.Max {
		delay = scaled
	}
	if b.Jitter > 0 {
		unit = min(max(unit, 0), 1)
		spread := time.Duration(float64(delay) * b.Jitter * unit)
		delay = min(delay+spread, b.Max)
	}
	return delay
}

// IdleInterval is a periodic controller's interval: it doubles while a tick
// finds no work, clamps at Max, and resets to Base when a tick finds work.
type IdleInterval struct {
	Base    time.Duration
	Max     time.Duration
	current time.Duration
}

// NewIdleInterval starts a controller at base, clamped by max.
func NewIdleInterval(base, max time.Duration) (*IdleInterval, error) {
	if base <= 0 || max < base {
		return nil, errors.New("loop: idle interval needs a positive base at or below its maximum")
	}
	return &IdleInterval{Base: base, Max: max, current: base}, nil
}

// Next records whether the last tick found work and returns the interval
// until the next tick.
func (interval *IdleInterval) Next(foundWork bool) time.Duration {
	if interval == nil {
		return 0
	}
	if foundWork {
		interval.current = interval.Base
		return interval.current
	}
	interval.current = min(interval.current*2, interval.Max)
	return interval.current
}
