package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/protection"
)

func TestProtectionEvidenceContract(t *testing.T) {
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	configured, activated, err := protection.Verify(repo)
	if err != nil {
		t.Fatal(err)
	}
	if configured == "" || activated == "" {
		t.Fatalf("protection evidence is incomplete: configured=%q activated=%q", configured, activated)
	}
	gate := gateContext{repo: repo, stepEvidence: map[string]string{}}
	if skipped, err := gate.stepProtection(); err != nil || skipped || gate.stepEvidence["protection"] == "" {
		t.Fatalf("stepProtection() = (%v, %v), evidence=%q", skipped, err, gate.stepEvidence["protection"])
	}
	if _, err := os.Stat(filepath.Join(repo, ".github", "protection.json")); err != nil {
		t.Fatal(err)
	}
}
