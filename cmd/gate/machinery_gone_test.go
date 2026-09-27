package main

import (
	"os"
	"path/filepath"
	"testing"
)

// requireGone fails for each repository path that still exists.
func requireGone(t *testing.T, paths ...string) {
	t.Helper()
	for _, path := range paths {
		if _, err := os.Stat(filepath.Join("..", "..", filepath.FromSlash(path))); !os.IsNotExist(err) {
			t.Errorf("%s still exists (%v)", path, err)
		}
	}
}

// TestOldGateGone holds the landing path to the small gate: the old gate's
// package and its command modes are deleted.
func TestOldGateGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/gate")
}

// TestSelectionAnalyzersGone holds test selection to go test's own cache:
// the code manifest, its impact analysis and the check-planning package are
// deleted.
func TestSelectionAnalyzersGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/codemanifest", "internal/automationcheck", "cmd/code-manifest")
}

// TestCompletionAuthorityGone holds the plan to plain data: the completion
// authority, its transitions and its proof horizon are deleted, and git is
// the record of what landed.
func TestCompletionAuthorityGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/plan/completion.go", "internal/plan/projected_merge_authority.go",
		"internal/plan/completion_transitions.go", "internal/plan/proof_horizon.go", "docs/plan_proof_horizon.json")
}
