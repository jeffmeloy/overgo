// Package modelrecipe_test exercises the modelrecipe package's entry-authority
// boundaries from outside, the way a bypassing consumer would reach them.
package modelrecipe_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestProductionAuthorityBoundaries proves the model and routing entry
// authorities hold: the shared architecture-ratchet table carries both rules
// owned by this package, the live tree passes them, and a production file
// that constructs a populated prototype or routing decision directly is
// refused.
func TestProductionAuthorityBoundaries(t *testing.T) {
	rules := make([]closurescan.EntryAuthorityRule, 0, 2)
	for _, domain := range []closurescan.EntryAuthorityDomain{
		closurescan.EntryAuthorityModel, closurescan.EntryAuthorityRouting,
	} {
		rule, found := closurescan.EntryAuthorityRuleFor(domain)
		if !found || !strings.Contains(rule.Owner, "internal/modelrecipe") {
			t.Fatalf("%s entry authority rule = (found=%t, owner=%q)", domain, found, rule.Owner)
		}
		rules = append(rules, rule)
	}
	snapshot, err := repoanalysis.DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	report, err := closurescan.ValidateEntryAuthorities(snapshot, rules)
	if err != nil {
		t.Fatalf("live tree violates a modelrecipe entry authority: %v", err)
	}
	if len(report.Domains) != 2 || report.Domains[0].ProductionFiles == 0 {
		t.Fatalf("modelrecipe entry authorities inspected nothing: %+v", report)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\n" +
		"import \"overgo/internal/modelrecipe\"\n\n" +
		"func bypass() modelrecipe.RoutingDecision {\n" +
		"\treturn modelrecipe.RoutingDecision{Derivation: \"rogue\"}\n" +
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
		t.Fatalf("routing bypass was not refused: %v", err)
	}
}
