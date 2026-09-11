package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/codeprofile"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

// TestCatalogBindsInOneTransaction pins the one-transaction catalogue: one
// command takes a candidate's name and the reviewed text, proposes its
// coordinates, binds it and commits it, the store head moves by exactly
// one commit, the binding is active afterwards, and a repeated command
// refuses a candidate already catalogued without a commit. The staged
// surface is declared from the gate's own report of it and reloads under
// the gate's validation.
func TestCatalogBindsInOneTransaction(t *testing.T) {
	root := t.TempDir()
	relative := "internal/sample/policy.go"
	source := []byte("package sample\n\nconst PolicyWindow = 3 * 8\n")
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, source, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	text := catalogText{
		tier: string(closureledger.TierImplementation), understanding: "Fixed fixture policy.",
		closurePath: "Replace when the fixture contract changes.", rerankTrigger: "Fixture contract change.",
	}
	catalogued, rebound, err := catalogCandidates(root, "store", snapshot, []string{"PolicyWindow"}, text)
	if err != nil || catalogued != 1 || rebound != 0 {
		t.Fatalf("catalogue = %d catalogued, %d rebound, %v", catalogued, rebound, err)
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, sequence := store.Head(); sequence != 1 {
		t.Fatalf("the catalogue took %d commits, want one transaction", sequence)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	var candidate closurescan.Candidate
	for _, scanned := range candidates {
		if scanned.Name == "PolicyWindow" {
			candidate = scanned
		}
	}
	binding, err := candidate.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document, active, err := closureledger.ResolveActiveBinding(t.Context(), store, binding, candidate.ValueJSON())
	if err != nil || !active || document.Understanding != text.understanding {
		t.Fatalf("active document = (%+v, %t, %v)", document, active, err)
	}
	if _, _, err := catalogCandidates(root, "store", snapshot, []string{"PolicyWindow"}, text); err == nil || !strings.Contains(err.Error(), "not found by scan") {
		t.Fatalf("a catalogued candidate was catalogued again: %v", err)
	}
	if _, sequence := store.Head(); sequence != 1 {
		t.Fatalf("the refused catalogue committed: sequence %d", sequence)
	}
	if _, _, err := catalogCandidates(root, "store", snapshot, []string{"PolicyWindow"}, catalogText{tier: text.tier}); err == nil || !strings.Contains(err.Error(), "-understanding is required") {
		t.Fatalf("missing text accepted: %v", err)
	}

	docs := filepath.Join(root, "docs")
	if err := os.MkdirAll(docs, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(docs, "plan.json"), []byte(`{"items":[{"id":"item","status":"open","steps":[{"id":"do","status":"open"}]}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(docs, "staged_surface.json"), []byte(`{"version":2,"staged":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	added, err := stageSurface(root, "internal/sample/policy.go:Exported=zero,internal/sample/policy.go:Other=test-only", "Consumed by the fixture's next row.", "item/do")
	if err != nil || added != 2 {
		t.Fatalf("staged %d, %v", added, err)
	}
	declaration, err := codeprofile.LoadStagedSurface(filepath.Join(docs, "staged_surface.json"))
	if err != nil || len(declaration.Staged) != 2 || declaration.Staged[0].Package != "fixture/internal/sample" || declaration.Staged[0].Name != "Exported" || declaration.Staged[0].RetireWith != "item/do" || declaration.Version != artifact.SecondDocumentVersion {
		t.Fatalf("staged declaration = %+v, %v", declaration, err)
	}
	if added, err := stageSurface(root, "internal/sample/policy.go:Exported=zero", "Consumed by the fixture's next row.", "item/do"); err != nil || added != 0 {
		t.Fatalf("restaging = %d, %v; want nothing added", added, err)
	}
	if _, err := stageSurface(root, "internal/sample/policy.go:Later=zero", "", "item/do"); err == nil {
		t.Fatal("a staged entry without a reason was accepted")
	}
}
