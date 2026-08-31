// Package recipe_test exercises the Go-only runtime boundary for recipe
// definitions against the shared architecture-ratchet policy table.
package recipe_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestRSIRuntimeIsGoOnly proves recipes cannot smuggle non-Go capability into
// the runtime: the shared Go-only rule holds over the live tree, and a
// production file that loads a Go plugin — the classic non-Go extension door —
// is refused by the same rule the gate runs on every commit.
func TestRSIRuntimeIsGoOnly(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityGoOnly)
	if !found || !strings.Contains(rule.Owner, "compiled Go registrations") {
		t.Fatalf("go-only entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}
	snapshot, err := repoanalysis.DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(snapshot, []closurescan.EntryAuthorityRule{rule}); err != nil {
		t.Fatalf("live tree violates the go-only runtime authority: %v", err)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\nimport _ \"plugin\"\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("plugin loading was not refused: %v", err)
	}
}
