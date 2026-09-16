package worklease

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// WorkspaceClaimMode distinguishes read from write path access in the
// ordered view below; production compares claim lists pairwise.
type WorkspaceClaimMode string

const (
	// WorkspaceClaimRead claims shared read access to a path.
	WorkspaceClaimRead WorkspaceClaimMode = "read"
	// WorkspaceClaimWrite claims exclusive write access to a path.
	WorkspaceClaimWrite WorkspaceClaimMode = "write"
)

// WorkspaceClaim is one mode-qualified path in the deterministic
// acquisition order.
type WorkspaceClaim struct {
	Mode WorkspaceClaimMode
	Path string
}

// workspaceClaimOrder is the deterministic acquisition order the conflict
// test checks; production compares claims pairwise.
func workspaceClaimOrder(claims WorkspaceClaims) []WorkspaceClaim {
	if claims.WholeWorktree {
		return []WorkspaceClaim{{Mode: WorkspaceClaimWrite, Path: "*"}}
	}
	result := make([]WorkspaceClaim, 0, len(claims.Read)+len(claims.Write))
	for _, path := range claims.Read {
		result = append(result, WorkspaceClaim{Mode: WorkspaceClaimRead, Path: path})
	}
	for _, path := range claims.Write {
		result = append(result, WorkspaceClaim{Mode: WorkspaceClaimWrite, Path: path})
	}
	slices.SortFunc(result, func(left, right WorkspaceClaim) int {
		if order := strings.Compare(strings.ToLower(left.Path), strings.ToLower(right.Path)); order != 0 {
			return order
		}
		return strings.Compare(string(left.Mode), string(right.Mode))
	})
	return result
}

func TestWorkspaceClaims(t *testing.T) {
	root := t.TempDir()
	internal := filepath.Join(root, "internal")
	if err := os.Mkdir(internal, 0o755); err != nil {
		t.Fatal(err)
	}
	worktree, firstClaims := ResolveWorkspaceClaims(root, []string{"docs"}, []string{"internal"})
	_, secondClaims := ResolveWorkspaceClaims(root, nil, []string{"internal/agentloop"})
	first := leaseFixture(t, "first", worktree)
	first.Claims = firstClaims
	second := leaseFixture(t, "second", worktree)
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
	if !opaque.WholeWorktree || workspaceClaimOrder(opaque)[0].Path != "*" {
		t.Fatalf("opaque claims = %+v", opaque)
	}
	order := workspaceClaimOrder(firstClaims)
	if len(order) != 2 || order[0].Path > order[1].Path {
		t.Fatalf("claim order = %+v", order)
	}
}
