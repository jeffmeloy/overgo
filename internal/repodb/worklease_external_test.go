package repodb_test

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/plan"
	"overgo/internal/repodb"
)

func TestWorkLeaseCompareAndSet(t *testing.T) {
	newLease := func(task string) plan.WorkLease {
		lease, err := plan.NewWorkLease(plan.WorkLease{
			Task: task, Worktree: "C:/repo/lane", Branch: "codex/lane", Role: "developer",
			TargetHead: "0123456789abcdef0123456789abcdef01234567", DependsOn: []string{}, ConflictsWith: []string{},
			Resources: plan.ResourceRequest{CPUThreads: 8, HostRAMGiB: 16}, EvidenceLanes: []string{"go-test"}, ExpiresAt: "2026-08-15T00:00:00Z",
		})
		if err != nil {
			t.Fatal(err)
		}
		return lease
	}
	store, err := repodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	first := newLease("first/do")
	batch, _ := plan.WorkLeaseBatch(first, nil)
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	second := newLease("second/do")
	batch, _ = plan.WorkLeaseBatch(second, nil)
	if _, err := store.Commit(ctx, batch); !errors.Is(err, repodb.ErrAliasConflict) {
		t.Fatalf("unconditional lease takeover = %v", err)
	}
	wrong, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("wrong lease"))
	if err != nil {
		t.Fatal(err)
	}
	batch, _ = plan.WorkLeaseBatch(second, &wrong)
	if _, err := store.Commit(ctx, batch); !errors.Is(err, repodb.ErrAliasConflict) {
		t.Fatalf("lease takeover with wrong expected identity = %v", err)
	}
	batch, _ = plan.WorkLeaseBatch(second, &first.ID)
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	current, ok, err := store.ResolveAlias(ctx, plan.WorkLeaseAlias(first.Worktree))
	if err != nil || !ok || current != second.ID {
		t.Fatalf("current lease = (%s, %t, %v)", current, ok, err)
	}
}
