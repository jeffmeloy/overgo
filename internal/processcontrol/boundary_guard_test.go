// Package processcontrol_test exercises the process entry-authority boundary
// against the shared architecture-ratchet policy table.
package processcontrol_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestProductionAuthorityBoundaries is the process boundary: outside
// internal/processcontrol, production code neither spawns processes through
// os/exec nor kills them directly. The policy lives in the shared
// architecture-ratchet table the gate runs on every commit; this test
// exercises exactly that rule, so there is one policy owner and no copy.
// Grandfathered files are enumerated with retirement reasons and must still
// hold a site, so the exception list can only shrink.
func TestProductionAuthorityBoundaries(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityProcess)
	if !found || !strings.Contains(rule.Owner, "internal/processcontrol") {
		t.Fatalf("process entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}
	if len(rule.Exceptions) == 0 {
		t.Fatal("process boundary lost its enumerated grandfather list")
	}
	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\nimport \"os/exec\"\n\nfunc bypass() error {\n\treturn exec.Command(\"git\", \"status\").Run()\n}\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("direct exec bypass was not refused: %v", err)
	}
}
