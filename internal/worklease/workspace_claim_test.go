package worklease

import (
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
	worktree := "C:/repo/claims"
	claims := func(read, write []string) WorkspaceClaims {
		value := WorkspaceClaims{Read: read, Write: write}
		canonicalizeClaims(&value)
		return value
	}
	firstClaims := claims([]string{worktree + "/docs"}, []string{worktree + "/internal"})
	first := leaseFixture("first", worktree)
	first.Claims = firstClaims
	second := leaseFixture("second", worktree)
	second.Claims = claims(nil, []string{worktree + "/internal/agentloop"})
	if !WorkspaceClaimsConflict(first, second) {
		t.Fatal("nested write claims did not conflict")
	}
	readClaims := claims([]string{worktree + "/internal"}, nil)
	first.Claims, second.Claims = readClaims, readClaims
	if WorkspaceClaimsConflict(first, second) {
		t.Fatal("read-only claims conflicted")
	}
	opaque := claims(nil, nil)
	if !opaque.WholeWorktree || workspaceClaimOrder(opaque)[0].Path != "*" {
		t.Fatalf("opaque claims = %+v", opaque)
	}
	order := workspaceClaimOrder(firstClaims)
	if len(order) != 2 || order[0].Path > order[1].Path {
		t.Fatalf("claim order = %+v", order)
	}
}

func canonicalizeClaims(claims *WorkspaceClaims) {
	if claims == nil {
		return
	}
	for _, values := range []*[]string{&claims.Read, &claims.Write} {
		for index, value := range *values {
			(*values)[index] = filepath.ToSlash(filepath.Clean(value))
		}
		slices.SortFunc(*values, func(left, right string) int { return strings.Compare(strings.ToLower(left), strings.ToLower(right)) })
		*values = slices.CompactFunc(*values, strings.EqualFold)
	}
	if len(claims.Read)+len(claims.Write) == 0 {
		claims.WholeWorktree = true
	}
	if claims.WholeWorktree {
		claims.Read, claims.Write = nil, nil
	}
}
