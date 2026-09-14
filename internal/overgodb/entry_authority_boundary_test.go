package overgodb_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestProductionAuthorityBoundaries proves the storage entry authority holds:
// the shared architecture-ratchet table carries a storage rule owned by this
// package, the live tree passes it, and a production file that reaches for
// the journal layout directly is refused, as is a grandfathered exception
// whose sites have disappeared.
func TestProductionAuthorityBoundaries(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityStorage)
	if !found || !strings.Contains(rule.Owner, "internal/overgodb") {
		t.Fatalf("storage entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\nimport \"os\"\n\nfunc bypass() error {\n\treturn os.WriteFile(\"overgodb.log\", nil, 0o644)\n}\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("journal bypass was not refused: %v", err)
	}

	stale := rule
	stale.Exceptions = map[string]string{"internal/rogue/gone.go": "site no longer exists"}
	empty := t.TempDir()
	if err := os.MkdirAll(filepath.Join(empty, "internal", "clean"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(empty, "internal", "clean", "clean.go"), []byte("package clean\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cleanSnapshot, err := repoanalysis.DiscoverGo(empty, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(cleanSnapshot, []closurescan.EntryAuthorityRule{stale}); err == nil ||
		!strings.Contains(err.Error(), "no longer holds a direct site") {
		t.Fatalf("stale storage exception was not refused: %v", err)
	}
}
