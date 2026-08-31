// Package agenttool_test exercises the tool entry-authority boundary against
// the shared architecture-ratchet policy table.
package agenttool_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestProductionAuthorityBoundaries proves the tool entry authority holds:
// manuals enter through registration and resolve from the store catalog, so
// production code outside internal/agenttool constructing a manual by hand —
// a tool identity that never passed registration — is refused by the shared
// architecture-ratchet rule the gate runs on every commit.
func TestProductionAuthorityBoundaries(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityTool)
	if !found || !strings.Contains(rule.Owner, "internal/agenttool") {
		t.Fatalf("tool entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}
	snapshot, err := repoanalysis.DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(snapshot, []closurescan.EntryAuthorityRule{rule}); err != nil {
		t.Fatalf("live tree violates the tool entry authority: %v", err)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\nimport \"overgo/internal/agenttool\"\n\nfunc bypass() agenttool.Manual {\n\treturn agenttool.Manual{Name: \"unregistered\"}\n}\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("hand-built manual was not refused: %v", err)
	}
}
