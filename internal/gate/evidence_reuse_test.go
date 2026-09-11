package gate

import (
	"os"
	"path/filepath"
	"testing"
)

// TestGateReusesUnchangedCheckEvidence pins the attempt-evidence-reuse
// contract on the gate side: every deterministic check with an exact input
// fingerprint participates in cross-attempt reuse, store-coupled checks and
// the commit and device checks never do, and a check's fingerprint moves
// exactly with its owned inputs so a narrow fix repays only the invalidated
// checks.
func TestGateReusesUnchangedCheckEvidence(t *testing.T) {
	for phase, reusable := range map[string]bool{
		"fmt": true, "style": true, "profile": true, "scope": true, "protection": true,
		"manifest": true, "sbom": true, "claims": true, "docs": true, "architecture": true,
		"vet": true, "build": true, "test": true, "test-owners": true, "test-device": true,
		"magics": false, "acceptance": false, "published": false, "device": false, "commit": false,
	} {
		if got := phaseReusesEvidence(phase); got != reusable {
			t.Fatalf("phaseReusesEvidence(%q) = %t", phase, got)
		}
	}
	for _, check := range (&gateContext{}).pipelineChecks() {
		name := check.Descriptor.Name
		if phaseReusesEvidence(name) && !phaseOwnsPath(name, "internal/plan/plan.go") &&
			!phaseOwnsPath(name, "docs/plan.json") && !phaseOwnsPath(name, "scripts/guard.sh") {
			t.Fatalf("reusable check %q owns no input surface", name)
		}
	}

	repo := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repo, "internal", "unit"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(repo, "internal", "unit", "unit.go")
	if err := os.WriteFile(source, []byte("package unit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	document := filepath.Join(repo, "docs", "NOTES.md")
	if err := os.WriteFile(document, []byte("notes\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	paths := []string{"internal/unit/unit.go", "docs/NOTES.md"}
	styleBefore, err := fingerprintPhaseInputs(repo, "style", paths)
	if err != nil {
		t.Fatal(err)
	}
	docsBefore, err := fingerprintPhaseInputs(repo, "docs", paths)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(document, []byte("edited notes\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	styleAfterDocs, err := fingerprintPhaseInputs(repo, "style", paths)
	if err != nil {
		t.Fatal(err)
	}
	docsAfterDocs, err := fingerprintPhaseInputs(repo, "docs", paths)
	if err != nil {
		t.Fatal(err)
	}
	if styleAfterDocs != styleBefore {
		t.Fatal("a documentation edit invalidated the style check's input fingerprint")
	}
	if docsAfterDocs == docsBefore {
		t.Fatal("a documentation edit left the docs check's input fingerprint unchanged")
	}
	if err := os.WriteFile(source, []byte("package unit\n\nvar edited = true\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	styleAfterSource, err := fingerprintPhaseInputs(repo, "style", paths)
	if err != nil {
		t.Fatal(err)
	}
	if styleAfterSource == styleBefore {
		t.Fatal("a source edit left the style check's input fingerprint unchanged")
	}
}
