package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestArchitectureRatchetIncludesGoOnlyPolicy pins that the same
// always-required architecture check carries the Go-only runtime rule: the
// shared policy table declares the domain, the gate demands it on every
// commit, the live tree satisfies it, and a production runtime-script literal
// is refused. No second guard process or policy owner is introduced.
func TestArchitectureRatchetIncludesGoOnlyPolicy(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityGoOnly)
	if !found || !strings.Contains(rule.Owner, "Go") {
		t.Fatalf("go-only entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}
	gate := &gateContext{repo: filepath.Join("..", ".."), paths: []string{"README.md"}}
	if skipped, err := gate.stepArchitecture(); err != nil || skipped {
		t.Fatalf("architecture ratchet over the live tree = (skipped=%t, %v)", skipped, err)
	}
	governed := false
	for _, line := range gate.honesty {
		if strings.HasPrefix(line, "entry authority ratchet:") && strings.Contains(line, "go-only") {
			governed = true
		}
	}
	if !governed {
		t.Fatalf("gate ratchet does not report go-only governance: %q", gate.honesty)
	}

	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package rogue\n\nconst deployHook = \"deploy.ps1\"\n"
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{rule}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("runtime script literal was not refused: %v", err)
	}
}
