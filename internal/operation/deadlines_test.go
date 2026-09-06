package operation

import (
	"context"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// deadlineClock: a settable manager clock so reports and flushes carry
// exact instants.
type deadlineClock struct{ now time.Time }

func (clock *deadlineClock) read() time.Time { return clock.now }

// deadlineOperation: an executor that blocks until cancelled and reports
// progress once per message.
func deadlineOperation(t *testing.T, manager *Manager, name string, deadlines *Deadlines) (artifact.ID, chan<- struct{}) {
	t.Helper()
	report := make(chan struct{}, 8)
	started := make(chan struct{})
	id, err := manager.Submit(t.Context(), Request{
		Task: recipe.TaskGeneration, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, name), Deadlines: deadlines,
	}, func(ctx context.Context, reporter Reporter) (Completion, error) {
		close(started)
		progress := uint64(0)
		for {
			select {
			case <-ctx.Done():
				return Completion{Run: testutil.ArtifactID(t, artifact.KindRun, name)}, ctx.Err()
			case <-report:
				progress++
				reporter.Progress(progress, nil)
			}
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	return id, report
}

// reportAt: one progress report carrying the clock instant, applied before return.
func reportAt(t *testing.T, manager *Manager, clock *deadlineClock, id artifact.ID, report chan<- struct{}, at time.Time, completed uint64) {
	t.Helper()
	clock.now = at
	report <- struct{}{}
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if status, _ := manager.Status(id); status.Progress.Completed >= completed {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("operation %s never reported progress %d", id, completed)
}

func mustStatus(t *testing.T, manager *Manager, id artifact.ID) Status {
	t.Helper()
	status, ok := manager.Status(id)
	if !ok {
		t.Fatalf("operation %s is unknown", id)
	}
	return status
}

// TestScheduleAndExecutionDeadlines pins: invalid bounds are refused at
// submission; an operation that never starts is cancelled when its schedule
// deadline elapses; one that started is cancelled when its execution
// deadline elapses without a further report; a later report refreshes the
// execution deadline once flushed; every cancellation records its reason in
// the status and the deadline state; an operation without deadlines is
// never cancelled by the flush.
func TestScheduleAndExecutionDeadlines(t *testing.T) {
	manager := newTestManager(t)
	clock := &deadlineClock{now: time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)}
	manager.now = clock.read
	deadlines := &Deadlines{Schedule: 10 * time.Second, Execution: 30 * time.Second, Ceiling: 2 * time.Minute}
	if _, err := manager.Submit(t.Context(), Request{
		Task: recipe.TaskGeneration, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "bad"),
		Deadlines: &Deadlines{Schedule: time.Second, Execution: time.Minute, Ceiling: time.Second},
	}, func(context.Context, Reporter) (Completion, error) { return Completion{}, nil }); err == nil {
		t.Fatal("a ceiling below the execution budget was admitted")
	}

	admitted := clock.now
	waiting, _ := deadlineOperation(t, manager, "waiting", deadlines)
	running, report := deadlineOperation(t, manager, "running", deadlines)
	unbounded, _ := deadlineOperation(t, manager, "unbounded", nil)
	if state := mustStatus(t, manager, waiting).Deadline; state == nil || !state.ScheduleBy.Equal(admitted.Add(10*time.Second)) || !state.CeilingAt.Equal(admitted.Add(2*time.Minute)) {
		t.Fatalf("admitted deadline state = %+v", state)
	}
	if manager.FlushDeadlines(admitted.Add(9*time.Second)) != 0 {
		t.Fatal("a deadline elapsed before its bound")
	}
	startedAt := admitted.Add(5 * time.Second)
	reportAt(t, manager, clock, running, report, startedAt, 1)
	if cancelled := manager.FlushDeadlines(admitted.Add(11 * time.Second)); cancelled != 1 {
		t.Fatalf("schedule flush cancelled %d, want the never-started operation only", cancelled)
	}
	status, err := manager.Wait(t.Context(), waiting)
	if err != nil || status.State != StateCancelled || !strings.Contains(status.Failure, "schedule deadline elapsed") || status.Deadline.Cancelled != status.Failure {
		t.Fatalf("waiting operation = %+v, %v", status, err)
	}
	started := mustStatus(t, manager, running).Deadline
	if !started.Started.Equal(startedAt) || !started.ExecuteBy.Equal(startedAt.Add(30*time.Second)) {
		t.Fatalf("started deadline state = %+v", started)
	}

	refreshAt := startedAt.Add(20 * time.Second)
	reportAt(t, manager, clock, running, report, refreshAt, 2)
	if buffered := mustStatus(t, manager, running).Deadline; buffered.Buffered != 1 || buffered.Refreshes != 0 {
		t.Fatalf("report was applied before the flush: %+v", buffered)
	}
	if manager.FlushDeadlines(startedAt.Add(31*time.Second)) != 0 {
		t.Fatal("a refreshed execution deadline elapsed at the original bound")
	}
	refreshed := mustStatus(t, manager, running).Deadline
	if refreshed.Refreshes != 1 || refreshed.Buffered != 0 || !refreshed.ExecuteBy.Equal(refreshAt.Add(30*time.Second)) {
		t.Fatalf("refreshed deadline state = %+v", refreshed)
	}
	if cancelled := manager.FlushDeadlines(refreshAt.Add(31 * time.Second)); cancelled != 1 {
		t.Fatalf("execution flush cancelled %d", cancelled)
	}
	status, err = manager.Wait(t.Context(), running)
	if err != nil || status.State != StateCancelled || !strings.Contains(status.Failure, "execution deadline elapsed") {
		t.Fatalf("running operation = %+v, %v", status, err)
	}
	if manager.FlushDeadlines(admitted.Add(24*time.Hour)) != 0 || mustStatus(t, manager, unbounded).State != StateRunning {
		t.Fatal("an operation without deadlines was cancelled by the flush")
	}
}
