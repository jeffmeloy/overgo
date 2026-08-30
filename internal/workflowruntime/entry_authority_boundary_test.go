// Package workflowruntime_test exercises the trigger entry-authority boundary
// against the shared architecture-ratchet policy table.
package workflowruntime_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestProductionAuthorityBoundaries proves the trigger entry authority holds:
// webhook deliveries are constructed only by the runrecord ledger and the
// dispatch authority, so production code elsewhere fabricating a delivery —
// an execution trigger that never passed admission — is refused by the shared
// architecture-ratchet rule the gate runs on every commit.
func TestProductionAuthorityBoundaries(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityTrigger)
	if !found || !strings.Contains(rule.Owner, "internal/workflowruntime") {
		t.Fatalf("trigger entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}
	snapshot, err := repoanalysis.DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(snapshot, []closurescan.EntryAuthorityRule{rule}); err != nil {
		t.Fatalf("live tree violates the trigger entry authority: %v", err)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\nimport \"overgo/internal/runrecord\"\n\nfunc bypass() runrecord.WebhookDelivery {\n\treturn runrecord.WebhookDelivery{}\n}\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("fabricated webhook delivery was not refused: %v", err)
	}
}
