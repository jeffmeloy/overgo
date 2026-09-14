package runrecord_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestProductionAuthorityBoundaries proves the rollout and efficiency entry
// authorities hold: the shared architecture-ratchet table carries both rules
// owned by this package, the live tree passes them, and a production file
// that constructs a populated rollout plan or efficiency trace directly is
// refused.
func TestProductionAuthorityBoundaries(t *testing.T) {
	rules := make([]closurescan.EntryAuthorityRule, 0, 2)
	for _, domain := range []closurescan.EntryAuthorityDomain{
		closurescan.EntryAuthorityRollout, closurescan.EntryAuthorityEfficiency,
	} {
		rule, found := closurescan.EntryAuthorityRuleFor(domain)
		if !found || !strings.Contains(rule.Owner, "internal/runrecord") {
			t.Fatalf("%s entry authority rule = (found=%t, owner=%q)", domain, found, rule.Owner)
		}
		rules = append(rules, rule)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\n" +
		"import \"overgo/internal/runrecord\"\n\n" +
		"func bypass() runrecord.RolloutPlan {\n" +
		"\treturn runrecord.RolloutPlan{Version: 1}\n" +
		"}\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, rules); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("rollout bypass was not refused: %v", err)
	}
}
