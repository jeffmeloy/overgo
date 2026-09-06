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

// TestProgressExtendsExecutionDeadline pins: several reports between flushes
// are buffered and applied as one refresh; a refresh never moves the
// execution deadline past the ceiling; a flush past the ceiling cancels with
// the reason and the refresh count recorded; an operator budget bounds the
// ceiling a request may declare.
func TestProgressExtendsExecutionDeadline(t *testing.T) {
	manager := newTestManager(t)
	clock := &deadlineClock{now: time.Date(2026, 9, 6, 15, 0, 0, 0, time.UTC)}
	manager.now = clock.read
	manager.BoundDeadlines(time.Hour)
	if _, err := manager.Submit(t.Context(), Request{
		Task: recipe.TaskGeneration, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "over-budget"),
		Deadlines: &Deadlines{Schedule: time.Minute, Execution: time.Minute, Ceiling: 2 * time.Hour},
	}, func(context.Context, Reporter) (Completion, error) { return Completion{}, nil }); err == nil || !strings.Contains(err.Error(), "operator budget") {
		t.Fatalf("a ceiling above the operator budget was admitted: %v", err)
	}

	admitted := clock.now
	deadlines := &Deadlines{Schedule: 10 * time.Second, Execution: 30 * time.Second, Ceiling: 70 * time.Second}
	id, report := deadlineOperation(t, manager, "refreshing", deadlines)
	startedAt := admitted.Add(5 * time.Second)
	reportAt(t, manager, clock, id, report, startedAt, 1)
	for step := range 3 {
		reportAt(t, manager, clock, id, report, startedAt.Add(time.Duration(step+1)*5*time.Second), uint64(step+2))
	}
	if state := mustStatus(t, manager, id).Deadline; state.Buffered != 3 || state.Refreshes != 0 {
		t.Fatalf("reports were not buffered: %+v", state)
	}
	if manager.FlushDeadlines(startedAt.Add(20*time.Second)) != 0 {
		t.Fatal("a flush cancelled a refreshing operation")
	}
	state := mustStatus(t, manager, id).Deadline
	if state.Refreshes != 1 || state.Buffered != 0 || !state.ExecuteBy.Equal(startedAt.Add(45*time.Second)) {
		t.Fatalf("buffered reports did not apply as one refresh: %+v", state)
	}

	// A refresh near the ceiling is clamped to it.
	reportAt(t, manager, clock, id, report, startedAt.Add(55*time.Second), 5)
	if manager.FlushDeadlines(startedAt.Add(56*time.Second)) != 0 {
		t.Fatal("a flush before the ceiling cancelled the operation")
	}
	state = mustStatus(t, manager, id).Deadline
	if state.Refreshes != 2 || !state.ExecuteBy.Equal(admitted.Add(70*time.Second)) {
		t.Fatalf("refresh was not clamped to the ceiling: %+v", state)
	}

	// Progress past the ceiling extends nothing: the flush cancels.
	reportAt(t, manager, clock, id, report, admitted.Add(69*time.Second), 6)
	if cancelled := manager.FlushDeadlines(admitted.Add(70 * time.Second)); cancelled != 1 {
		t.Fatalf("ceiling flush cancelled %d", cancelled)
	}
	status, err := manager.Wait(t.Context(), id)
	if err != nil || status.State != StateCancelled || !strings.Contains(status.Failure, "execution ceiling reached") || !strings.Contains(status.Failure, "3 refreshes") {
		t.Fatalf("operation past its ceiling = %+v, %v", status, err)
	}
}
