package plan

import (
	"bytes"
	"fmt"
	"os"
	"overgo/internal/authoritylock"
	"overgo/internal/worklease"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

const orchestrationHead = "0123456789abcdef0123456789abcdef01234567"

func TestStopStateContract(t *testing.T) {
	t.Parallel()
	fixture := func(t *testing.T) (string, *overgodb.Store, ControlEvent) {
		t.Helper()
		root := t.TempDir()
		store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { store.Close() })
		return root, store, ControlEvent{Kind: "stop", Lane: worklease.UnassignedRole, Worker: "stop-worker", Worktree: filepath.ToSlash(root), Mode: ExecutionAll, ReasonCode: "user-stop", Detail: "operator asked to stop", CodeCommit: orchestrationHead}
	}
	record := func(t *testing.T, store *overgodb.Store, event ControlEvent) ControlEvent {
		t.Helper()
		got, err := RecordControlEvent(t.Context(), store, event)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	t.Run("persistent scope and exact maintenance", func(t *testing.T) {
		root, store, event := fixture(t)
		stop := record(t, store, event)
		changed := strings.Repeat("b", len(orchestrationHead))
		current, err := ReadStop(t.Context(), store, root, changed, ExecutionInteractive)
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
		granted := record(t, store, grant)
		if err := current.RequireExecution(t.Context(), store, event.Lane, event.Worker, ExecutionInteractive, grant.Task, granted.ID.String()); err != nil {
			t.Fatal(err)
		}
		for _, mutate := range []func(*ControlEvent){func(e *ControlEvent) { e.Worker = "another-worker" }, func(e *ControlEvent) { e.Task = "row/other" }, func(e *ControlEvent) { e.Lane = "other-owner" }} {
			wrong := event
			wrong.Task = grant.Task
			mutate(&wrong)
			if err := current.RequireExecution(t.Context(), store, wrong.Lane, wrong.Worker, ExecutionInteractive, wrong.Task, granted.ID.String()); err == nil {
				t.Fatal("maintenance escaped owner/task scope")
			}
		}
		if err := current.RequireExecution(t.Context(), store, event.Lane, event.Worker, ExecutionUnattended, grant.Task, granted.ID.String()); err == nil {
			t.Fatal("maintenance resumed unattended work")
		}
		persisted, err := ReadStop(t.Context(), store, root, changed, ExecutionUnattended)
		if err != nil || !persisted.Blocked || persisted.ID != stop.ID {
			t.Fatalf("grant moved stop: %+v %v", persisted, err)
		}
		reopened, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
		if err != nil {
			t.Fatal(err)
		}
		defer reopened.Close()
		restarted, err := ReadStop(t.Context(), reopened, root, changed, ExecutionInteractive)
		if err != nil || restarted.ID != stop.ID || !restarted.Blocked {
			t.Fatal("restart lost stop")
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
			if _, err := RecordControlEvent(t.Context(), store, wrong); err == nil {
				t.Fatal("resume ignored owner/scope/identity mismatch")
			}
		}
		done := record(t, store, resumed)
		after, err := ReadStop(t.Context(), reopened, root, orchestrationHead, ExecutionUnattended)
		if err != nil || after.Blocked || after.State != "resumed" || after.ID != done.ID {
			t.Fatalf("explicit resume not visible to existing reader: %+v %v", after, err)
		}
		if err := after.RequireExecution(t.Context(), store, event.Lane, event.Worker, ExecutionInteractive, grant.Task, granted.ID.String()); err == nil {
			t.Fatal("stale maintenance survived explicit resume")
		}
	})
	t.Run("stop while gate mutation owner runs", func(t *testing.T) {
		root, store, event := fixture(t)
		lock, err := authoritylock.Acquire(root)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		record(t, store, event)
		status, err := ReadStop(t.Context(), store, root, event.CodeCommit, ExecutionUnattended)
		if err != nil || !status.Blocked {
			t.Fatal("running gate prevented stop")
		}
	})
	t.Run("scope leaves unrelated mode eligible", func(t *testing.T) {
		root, store, event := fixture(t)
		event.Mode = ExecutionUnattended
		record(t, store, event)
		for _, mode := range []string{ExecutionInteractive, ExecutionUnattended} {
			got, err := ReadStop(t.Context(), store, root, event.CodeCommit, mode)
			if err != nil || got.Blocked != (mode == ExecutionUnattended) {
				t.Fatalf("scope=%s %+v %v", mode, got, err)
			}
		}
	})
	t.Run("legacy mismatch migration and malformed repair", func(t *testing.T) {
		for _, malformed := range []bool{false, true} {
			t.Run(fmt.Sprint(malformed), func(t *testing.T) {
				root, store, event := fixture(t)
				if err := os.MkdirAll(filepath.Join(root, "docs"), 0700); err != nil {
					t.Fatal(err)
				}
				raw := fmt.Appendf(nil, "{\"reason\":\"user-stop: operator paused\",\"head\":%q}\n", orchestrationHead)
				if malformed {
					raw = []byte("{invalid stop bytes")
				}
				path := filepath.Join(root, legacyStopPath)
				if err := os.WriteFile(path, raw, 0600); err != nil {
					t.Fatal(err)
				}
				got, err := ReadStop(t.Context(), store, root, strings.Repeat("b", len(orchestrationHead)), ExecutionUnattended)
				if err != nil || !got.Blocked || (!malformed && !got.LegacyMismatch) {
					t.Fatalf("legacy projection: %+v %v", got, err)
				}
				event.Previous = got.ID
				if !malformed {
					event.Kind = "resume"
					event.ReasonCode = "operator-resume"
					event.Detail = "operator explicitly continued"
				}
				migrated := record(t, store, event)
				actual, err := os.ReadFile(path)
				if err != nil || !bytes.Equal(actual, raw) {
					t.Fatal("migration rewrote legacy marker")
				}
				content, found, err := artifact.ReadContent(t.Context(), store, got.ID)
				if err != nil || !found || !bytes.Equal(content.Data, raw) {
					t.Fatal("migration did not retain exact legacy bytes")
				}
				final, err := ReadStop(t.Context(), store, root, event.CodeCommit, ExecutionUnattended)
				if err != nil || final.ID != migrated.ID || final.Blocked != malformed {
					t.Fatalf("migration state: %+v %v", final, err)
				}
			})
		}
	})
	t.Run("concurrent writers cannot replace unseen stop", func(t *testing.T) {
		_, store, event := fixture(t)
		start := make(chan struct{})
		results := make(chan error, 2)
		for _, worker := range []string{"worker-one", "worker-two"} {
			go func() {
				next := event
				next.Worker = worker
				<-start
				_, err := RecordControlEvent(t.Context(), store, next)
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
