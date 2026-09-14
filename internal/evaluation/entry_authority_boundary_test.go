// Package evaluation_test exercises the evaluation package's entry-authority
// boundary from outside, the way a bypassing consumer would reach it.
package evaluation_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestProductionAuthorityBoundaries proves the live-safety entry authority
// holds: the shared architecture-ratchet table carries a safety rule owned by
// this package, the live tree passes it, and a production file that
// constructs a populated live safety window directly is refused.
func TestProductionAuthorityBoundaries(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthoritySafety)
	if !found || !strings.Contains(rule.Owner, "internal/evaluation") {
		t.Fatalf("safety entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\n" +
		"import \"overgo/internal/evaluation\"\n\n" +
		"func bypass() evaluation.LiveSafetyWindow {\n" +
		"\treturn evaluation.LiveSafetyWindow{Version: 1}\n" +
		"}\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("safety bypass was not refused: %v", err)
	}
}
