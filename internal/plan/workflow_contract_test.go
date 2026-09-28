package plan

import (
	"os"
	"overgo/internal/authoritylock"
	"overgo/internal/worklease"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/processlock"
)

const orchestrationHead = "0123456789abcdef0123456789abcdef01234567"

func stopFixture(t *testing.T) (string, ControlEvent) {
	t.Helper()
	root := t.TempDir()
	return root, ControlEvent{Kind: "stop", Lane: worklease.UnassignedRole, Worker: "stop-worker", Worktree: filepath.ToSlash(root), Mode: ExecutionAll, ReasonCode: "user-stop", Detail: "operator asked to stop", CodeCommit: orchestrationHead}
}

func recordStop(t *testing.T, event ControlEvent) ControlEvent {
	t.Helper()
	got, err := RecordStop(t.Context(), event)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func TestStopStateContract(t *testing.T) {
	t.Parallel()
	t.Run("persistent scope and exact maintenance", func(t *testing.T) {
		root, event := stopFixture(t)
		stop := recordStop(t, event)
		changed := strings.Repeat("b", len(orchestrationHead))
		current, err := ReadStop(t.Context(), root, changed, ExecutionInteractive)
		if err != nil || !current.Blocked || current.ID != stop.ID {
			t.Fatalf("foreign commit resumed stop: %+v %v", current, err)
		}
		grant := event
		grant.Kind = "maintenance"
		grant.ReasonCode = "operator-maintenance"
		grant.Detail = "publish the stop repair"
		grant.Task = "row/do"
		grant.Previous = stop.ID
		grant.CodeCommit = changed
		granted := recordStop(t, grant)
		current, err = ReadStop(t.Context(), root, changed, ExecutionInteractive)
		if err != nil {
			t.Fatal(err)
		}
		if err := current.RequireExecution(event.Lane, event.Worker, ExecutionInteractive, grant.Task, granted.ID.String()); err != nil {
			t.Fatal(err)
		}
		for _, mutate := range []func(*ControlEvent){func(e *ControlEvent) { e.Worker = "another-worker" }, func(e *ControlEvent) { e.Task = "row/other" }, func(e *ControlEvent) { e.Lane = "other-owner" }} {
			wrong := event
			wrong.Task = grant.Task
			mutate(&wrong)
			if err := current.RequireExecution(wrong.Lane, wrong.Worker, ExecutionInteractive, wrong.Task, granted.ID.String()); err == nil {
				t.Fatal("maintenance escaped owner/task scope")
			}
		}
		if err := current.RequireExecution(event.Lane, event.Worker, ExecutionUnattended, grant.Task, granted.ID.String()); err == nil {
			t.Fatal("maintenance resumed unattended work")
		}
		persisted, err := ReadStop(t.Context(), root, changed, ExecutionUnattended)
		if err != nil || !persisted.Blocked || persisted.ID != stop.ID {
			t.Fatalf("grant moved stop: %+v %v", persisted, err)
		}
		resumed := event
		resumed.Kind = "resume"
		resumed.ReasonCode = "operator-resume"
		resumed.Detail = "operator continued"
		resumed.Previous = stop.ID
		resumed.CodeCommit = changed
		for _, mutate := range []func(*ControlEvent){func(e *ControlEvent) { e.Lane = "other-owner" }, func(e *ControlEvent) { e.Mode = ExecutionUnattended }, func(e *ControlEvent) { e.Previous = granted.ID }} {
			wrong := resumed
			mutate(&wrong)
			if _, err := RecordStop(t.Context(), wrong); err == nil {
				t.Fatal("resume ignored owner/scope/identity mismatch")
			}
		}
		done := recordStop(t, resumed)
		after, err := ReadStop(t.Context(), root, orchestrationHead, ExecutionUnattended)
		if err != nil || after.Blocked || after.State != "resumed" || after.ID != done.ID {
			t.Fatalf("explicit resume not visible: %+v %v", after, err)
		}
		if err := after.RequireExecution(event.Lane, event.Worker, ExecutionInteractive, grant.Task, granted.ID.String()); err == nil {
			t.Fatal("stale maintenance survived explicit resume")
		}
	})
	t.Run("stop while gate mutation owner runs", func(t *testing.T) {
		root, event := stopFixture(t)
		lock, err := authoritylock.Acquire(root)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		recordStop(t, event)
		status, err := ReadStop(t.Context(), root, event.CodeCommit, ExecutionUnattended)
		if err != nil || !status.Blocked {
			t.Fatal("running gate prevented stop")
		}
	})
	t.Run("scope leaves unrelated mode eligible", func(t *testing.T) {
		root, event := stopFixture(t)
		event.Mode = ExecutionUnattended
		recordStop(t, event)
		for _, mode := range []string{ExecutionInteractive, ExecutionUnattended} {
			got, err := ReadStop(t.Context(), root, event.CodeCommit, mode)
			if err != nil || got.Blocked != (mode == ExecutionUnattended) {
				t.Fatalf("scope=%s %+v %v", mode, got, err)
			}
		}
	})
	t.Run("malformed stop file blocks for repair", func(t *testing.T) {
		root, _ := stopFixture(t)
		if err := processlock.EnsureStateDirectory(root); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, processlock.StateDirectory, stopFile), []byte("{invalid stop bytes"), 0o600); err != nil {
			t.Fatal(err)
		}
		got, err := ReadStop(t.Context(), root, orchestrationHead, ExecutionUnattended)
		if err != nil || !got.Blocked || got.State != "invalid" {
			t.Fatalf("malformed stop file: %+v %v", got, err)
		}
	})
	t.Run("concurrent writers cannot replace unseen stop", func(t *testing.T) {
		_, event := stopFixture(t)
		start := make(chan struct{})
		results := make(chan error, 2)
		for _, worker := range []string{"worker-one", "worker-two"} {
			go func() {
				next := event
				next.Worker = worker
				<-start
				_, err := RecordStop(t.Context(), next)
				results <- err
			}()
		}
		close(start)
		wins := 0
		for range 2 {
			if <-results == nil {
				wins++
			}
		}
		if wins != 1 {
			t.Fatalf("stop winners=%d", wins)
		}
	})
}

// TestStopStateLivesOutsideTheStore records a stop and its resume in a
// checkout that holds a store: the decisions land in the checkout's process
// state and the store's head does not move, so an operator toggle neither
// churns the evidence every store-reading suite depends on nor needs the
// store to be read.
func TestStopStateLivesOutsideTheStore(t *testing.T) {
	t.Parallel()
	root, event := stopFixture(t)
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	before, sequence := store.Head()
	stop := recordStop(t, event)
	resume := event
	resume.Kind, resume.ReasonCode, resume.Detail, resume.Previous = "resume", "operator-resume", "operator continued", stop.ID
	recordStop(t, resume)
	if after, afterSequence := store.Head(); after != before || afterSequence != sequence {
		t.Fatalf("recording a stop moved the store head from %v/%d to %v/%d", before, sequence, after, afterSequence)
	}
	if _, err := os.Stat(filepath.Join(root, processlock.StateDirectory, stopFile)); err != nil {
		t.Fatalf("stop state is not in the checkout's process state: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(root, "overgodb-store")); err != nil {
		t.Fatal(err)
	}
	status, err := ReadStop(t.Context(), root, orchestrationHead, ExecutionUnattended)
	if err != nil || status.State != "resumed" || status.Blocked {
		t.Fatalf("stop state unreadable without the store: %+v %v", status, err)
	}
}
