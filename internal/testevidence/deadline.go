package testevidence

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

type testClock struct {
	remaining time.Duration
	started   time.Time
}

type testDeadline struct {
	mu      sync.Mutex
	budget  time.Duration
	idle    time.Time
	tests   map[string]testClock
	timer   *time.Timer
	cancel  context.CancelCauseFunc
	stopped bool
}

func newTestDeadline(budget time.Duration, cancel context.CancelCauseFunc) *testDeadline {
	if budget == 0 {
		return nil
	}
	d := &testDeadline{budget: budget, idle: time.Now().Add(budget), tests: map[string]testClock{}, cancel: cancel}
	d.mu.Lock()
	d.timer = time.AfterFunc(budget, d.expire)
	d.mu.Unlock()
	return d
}

func (d *testDeadline) stop() {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.stopped = true
	d.timer.Stop()
}

func (d *testDeadline) observe(event goTestEvent) {
	if d == nil || event.Action == "output" || strings.Contains(event.Test, "/") {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.stopped {
		return
	}
	now := time.Now()
	if d.arm(now) {
		return
	}
	d.advance(event, now)
	d.arm(now)
}

// advance consumes only lifecycle events. Output and subtest chatter cannot
// renew a running parent's budget; pause/cont preserve its remaining time.
func (d *testDeadline) advance(event goTestEvent, now time.Time) {
	if event.Package == "" || strings.Contains(event.Test, "/") {
		return
	}
	key := event.Package + ": " + event.Test
	clock, exists := d.tests[key]
	switch event.Action {
	case "start":
		if event.Test != "" {
			return
		}
	case "run":
		if event.Test == "" || exists {
			return
		}
		d.tests[key] = testClock{remaining: d.budget, started: now}
	case "pause":
		if !exists || clock.started.IsZero() {
			return
		}
		clock.remaining -= now.Sub(clock.started)
		clock.started = time.Time{}
		d.tests[key] = clock
	case "cont":
		if !exists || !clock.started.IsZero() {
			return
		}
		clock.started = now
		d.tests[key] = clock
	case "pass", "fail", "skip":
		delete(d.tests, key)
	default:
		return
	}
	d.idle = now.Add(d.budget)
}

func (d *testDeadline) next() (time.Time, string) {
	when, name := d.idle, "test command made no top-level progress"
	for key, clock := range d.tests {
		if !clock.started.IsZero() {
			deadline := clock.started.Add(clock.remaining)
			if deadline.Before(when) || deadline.Equal(when) && key < name {
				when, name = deadline, key+" exhausted active test time"
			}
		}
	}
	return when, name
}

// arm runs under mu, including timer callbacks that raced a lifecycle event.
func (d *testDeadline) arm(now time.Time) bool {
	when, name := d.next()
	if !now.Before(when) {
		d.stopped = true
		d.cancel(fmt.Errorf("%s (budget %s): %w", name, d.budget, context.DeadlineExceeded))
		d.timer.Stop()
		return true
	}
	d.timer.Reset(when.Sub(now))
	return false
}

func (d *testDeadline) expire() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if !d.stopped {
		d.arm(time.Now())
	}
}
