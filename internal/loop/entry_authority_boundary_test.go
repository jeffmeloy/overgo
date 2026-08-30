package loop_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestProductionAuthorityBoundaries proves the promotion entry authority the
// loop's evidence gates feed into holds: the recipe.active alias namespace
// changes only through the modelrecipe lifecycle, so production code
// elsewhere naming that namespace directly — a promotion that never passed
// its evidence gate — is refused by the shared architecture-ratchet rule the
// gate runs on every commit.
func TestProductionAuthorityBoundaries(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityPromotion)
	if !found || !strings.Contains(rule.Owner, "internal/modelrecipe") {
		t.Fatalf("promotion entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}
	snapshot, err := repoanalysis.DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(snapshot, []closurescan.EntryAuthorityRule{rule}); err != nil {
		t.Fatalf("live tree violates the promotion entry authority: %v", err)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\nfunc bypass() string {\n\treturn \"recipe.active.\" + \"model/task\"\n}\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("direct activation alias write was not refused: %v", err)
	}
}
