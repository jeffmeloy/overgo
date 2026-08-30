package inference_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestProductionAuthorityBoundaries proves the capability entry authority
// holds: a serving runner exists only through OpenWithProgram over a resolved
// active recipe, so production code outside internal/inference constructing a
// Runner literal — a serving capability that never passed activation — is
// refused by the shared architecture-ratchet rule the gate runs on every
// commit.
func TestProductionAuthorityBoundaries(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityCapability)
	if !found || !strings.Contains(rule.Owner, "internal/inference") {
		t.Fatalf("capability entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}
	snapshot, err := repoanalysis.DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(snapshot, []closurescan.EntryAuthorityRule{rule}); err != nil {
		t.Fatalf("live tree violates the capability entry authority: %v", err)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\nimport \"overgo/internal/inference\"\n\nfunc bypass() *inference.Runner {\n\treturn &inference.Runner{}\n}\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("hand-built runner was not refused: %v", err)
	}
}
