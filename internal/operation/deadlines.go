// Package operation admits, runs, bounds and records long-running
// operations behind the serving surfaces.
package operation

import (
	"errors"
	"fmt"
	"time"
)

// deadlineFlushInterval bounds how long a buffered refresh or an elapsed
// deadline waits before the flush applies it.
const deadlineFlushInterval = 5 * time.Second

// Deadlines are the two-phase bounds of one operation. Schedule bounds the wait
// from admission to the first progress report; Execution is the budget from
// that start, refreshed by each later progress report; Ceiling is the
// absolute bound from admission that no refresh exceeds.
type Deadlines struct {
	Schedule  time.Duration `json:"schedule"`
	Execution time.Duration `json:"execution"`
	Ceiling   time.Duration `json:"ceiling"`
}

// DeadlineState records the instants of one operation's deadlines and the
// refreshes applied; Cancelled carries the reason once a deadline elapsed.
type DeadlineState struct {
	Admitted   time.Time `json:"admitted"`
	Started    time.Time `json:"started,omitzero"`
	ScheduleBy time.Time `json:"schedule_by"`
	ExecuteBy  time.Time `json:"execute_by,omitzero"`
	CeilingAt  time.Time `json:"ceiling_at"`
	Refreshes  int       `json:"refreshes"`
	Buffered   int       `json:"buffered,omitzero"`
	Cancelled  string    `json:"cancelled,omitzero"`
}

// Validate requires every bound positive and the ceiling at least the execution budget.
func (d Deadlines) Validate() error {
	if d.Schedule <= 0 || d.Execution <= 0 || d.Ceiling <= 0 {
		return errors.New("operation: every deadline must be positive")
	}
	if d.Ceiling < d.Execution {
		return errors.New("operation: the ceiling is below the execution budget")
	}
	return nil
}

// newDeadlineState: instants from admission.
func newDeadlineState(deadlines Deadlines, admitted time.Time) *DeadlineState {
	return &DeadlineState{Admitted: admitted, ScheduleBy: admitted.Add(deadlines.Schedule), CeilingAt: admitted.Add(deadlines.Ceiling)}
}

// deadlineTimer: per-entry deadline bookkeeping; refresh requests are
// buffered here and applied by the flush.
type deadlineTimer struct {
	deadlines Deadlines
	state     DeadlineState
	pending   time.Time
}

// report: a progress report at now; the first starts execution, later ones
// buffer a refresh applied by the next flush.
func (timer *deadlineTimer) report(now time.Time) {
	if timer == nil || timer.state.Cancelled != "" {
		return
	}
	if timer.state.Started.IsZero() {
		timer.state.Started = now
		timer.state.ExecuteBy = minTime(now.Add(timer.deadlines.Execution), timer.state.CeilingAt)
		return
	}
	timer.pending = now
	timer.state.Buffered++
}

// flush: applies the buffered refresh, then decides: schedule elapsed
// before a start, execution elapsed, or ceiling reached; returns the reason.
func (timer *deadlineTimer) flush(now time.Time) string {
	if timer == nil || timer.state.Cancelled != "" {
		return ""
	}
	if !timer.pending.IsZero() {
		refreshed := timer.pending.Add(timer.deadlines.Execution)
		timer.pending = time.Time{}
		timer.state.Buffered = 0
		timer.state.Refreshes++
		if refreshed.After(timer.state.CeilingAt) {
			refreshed = timer.state.CeilingAt
		}
		if refreshed.After(timer.state.ExecuteBy) {
			timer.state.ExecuteBy = refreshed
		}
	}
	switch {
	case timer.state.Started.IsZero() && !now.Before(timer.state.ScheduleBy):
		timer.state.Cancelled = fmt.Sprintf("schedule deadline elapsed: no progress within %s of admission", timer.deadlines.Schedule)
	case !timer.state.Started.IsZero() && !now.Before(timer.state.CeilingAt):
		timer.state.Cancelled = fmt.Sprintf("execution ceiling reached: %s after admission, %d refreshes applied", timer.deadlines.Ceiling, timer.state.Refreshes)
	case !timer.state.Started.IsZero() && !now.Before(timer.state.ExecuteBy):
		timer.state.Cancelled = fmt.Sprintf("execution deadline elapsed: no progress within %s of the last report", timer.deadlines.Execution)
	}
	return timer.state.Cancelled
}

func minTime(left, right time.Time) time.Time {
	if left.Before(right) {
		return left
	}
	return right
}

// FlushDeadlines applies buffered refreshes and cancels every active
// operation whose deadline elapsed at now, recording the reason; returns the
// cancelled identities' count.
func (manager *Manager) FlushDeadlines(now time.Time) int {
	if manager == nil {
		return 0
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	cancelled := 0
	for _, current := range manager.entries {
		if current.deadline == nil || terminal(current.status.State) {
			continue
		}
		reason := current.deadline.flush(now)
		current.status.Deadline = deadlineStatus(current.deadline)
		if reason == "" {
			continue
		}
		current.status.Failure = reason
		if current.cancel != nil {
			current.cancel()
		}
		cancelled++
		manager.publishLocked(current.status)
	}
	return cancelled
}

// deadlineStatus: the published copy of the timer state.
func deadlineStatus(timer *deadlineTimer) *DeadlineState {
	if timer == nil {
		return nil
	}
	return new(timer.state)
}

// deadlineLoop: the flush at a bounded interval until the manager closes.
func (manager *Manager) deadlineLoop(interval time.Duration) {
	ticker := time.Tick(interval)
	for {
		select {
		case <-manager.stop:
			return
		case now := <-ticker:
			manager.FlushDeadlines(now)
		}
	}
}
