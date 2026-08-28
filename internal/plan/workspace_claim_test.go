package plan

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWorkspaceClaims(t *testing.T) {
	root := t.TempDir()
	internal := filepath.Join(root, "internal")
	if err := os.Mkdir(internal, 0o755); err != nil {
		t.Fatal(err)
	}
	worktree, firstClaims := ResolveWorkspaceClaims(root, []string{"docs"}, []string{"internal"})
	_, secondClaims := ResolveWorkspaceClaims(root, nil, []string{"internal/agentloop"})
	first := workLeaseFixture(t, "first", worktree)
	first.Claims = firstClaims
	second := workLeaseFixture(t, "second", worktree)
	second.Claims = secondClaims
	if !WorkspaceClaimsConflict(first, second) {
		t.Fatal("nested write claims did not conflict")
	}
	_, readClaims := ResolveWorkspaceClaims(root, []string{"internal"}, nil)
	first.Claims, second.Claims = readClaims, readClaims
	if WorkspaceClaimsConflict(first, second) {
		t.Fatal("read-only claims conflicted")
	}
	_, opaque := ResolveWorkspaceClaims(filepath.Join(root, "missing"), nil, nil)
	if !opaque.WholeWorktree || WorkspaceClaimOrder(opaque)[0].Path != "*" {
		t.Fatalf("opaque claims = %+v", opaque)
	}
	order := WorkspaceClaimOrder(firstClaims)
	if len(order) != 2 || order[0].Path > order[1].Path {
		t.Fatalf("claim order = %+v", order)
	}
}
