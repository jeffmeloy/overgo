package gate

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/worklease"
)

// TestCommonOwnerConsolidationAcceptance proves the first duplicated automation
// responsibility is removed: the operation manager no longer runs its own
// workspace-claim admission. Its Request carries only Task and Recipe and its
// sole constructor is NewManager, so mutation write-overlap detection lives only
// in the work-lease owner the plan frontier already consumes. The retained owner
// still decides overlap, and an operation admits, runs and completes through the
// single constructor with no claim path of its own.
func TestCommonOwnerConsolidationAcceptance(t *testing.T) {
	t.Parallel()

	// The sole conflict owner: work-lease claims still decide write overlap.
	// operation delegated this rule here instead of duplicating it.
	const repo = "C:/repo"
	writer := worklease.Lease{Worktree: repo, Claims: worklease.WorkspaceClaims{Write: []string{repo + "/internal"}}}
	overlapping := worklease.Lease{Worktree: repo, Claims: worklease.WorkspaceClaims{Write: []string{repo + "/internal/gate"}}}
	disjoint := worklease.Lease{Worktree: repo, Claims: worklease.WorkspaceClaims{Write: []string{repo + "/cmd"}}}
	if !worklease.WorkspaceClaimsConflict(writer, overlapping) {
		t.Fatal("the retained conflict owner missed a write overlap")
	}
	if worklease.WorkspaceClaimsConflict(writer, disjoint) {
		t.Fatal("the retained conflict owner reported a false overlap")
	}

	// The operation manager admits and completes through the single constructor:
	// a Request of only Task and Recipe, no repository-backed claim path.
	manager, err := operation.NewManager(4)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	request := operation.Request{Task: recipe.TaskInference, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "recipe")}
	id, err := manager.Submit(t.Context(), request, func(context.Context, operation.Reporter) (operation.Completion, error) {
		return operation.Completion{Run: testutil.ArtifactID(t, artifact.KindRun, "run")}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	if status.State != operation.StateCompleted {
		t.Fatalf("operation state = %s, want completed", status.State)
	}
}
