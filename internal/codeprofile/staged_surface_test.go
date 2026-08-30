package codeprofile

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

const stagedSurfaceTestEvidence = "evidence:sha256:6a3465f7325358362cae92f2a7b584167fcc5a6c56068bba0f4748095eef6190"

// TestStagedSurfacePartition pins the acceptance contract: a declared
// package+name pair is accepted, everything else keeps blocking, an
// absent file stages nothing, and an incomplete or duplicate-name entry
// refuses to load.
func TestStagedSurfacePartition(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "staged_surface.json")
	declaration, err := LoadStagedSurface(path)
	if err != nil || len(declaration.Staged) != 0 {
		t.Fatalf("absent declaration = %+v, %v", declaration, err)
	}
	body := `{"version":2,"staged":[{"package":"overgo/internal/agentloop","name":"NewDelegatedCoordinator","reason":"agent orchestration API","retained":{"classification":"test-support","evidence":"` + stagedSurfaceTestEvidence + `"}}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	declaration, err = LoadStagedSurface(path)
	if err != nil || len(declaration.Staged) != 1 {
		t.Fatalf("declaration = %+v, %v", declaration, err)
	}
	unconsumed := []ConsumerDeclaration{
		{Package: "overgo/internal/agentloop", Name: "NewDelegatedCoordinator", Kind: "func"},
		{Package: "overgo/internal/agentloop", Name: "Undeclared", Kind: "func"},
	}
	accepted, blocking := PartitionStagedSurface(unconsumed, declaration)
	if len(accepted) != 1 || accepted[0].Name != "NewDelegatedCoordinator" ||
		len(blocking) != 1 || blocking[0].Name != "Undeclared" {
		t.Fatalf("partition = accepted %+v blocking %+v", accepted, blocking)
	}
	incomplete := `{"version":2,"staged":[{"package":"p","name":"N","reason":"","retained":{"classification":"test-support","evidence":"` + stagedSurfaceTestEvidence + `"}}]}`
	if err := os.WriteFile(path, []byte(incomplete), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadStagedSurface(path); err == nil {
		t.Fatal("incomplete staged entry accepted")
	}
	// Gate authority decodes strictly: a duplicate name would silently
	// last-wins and drop a reviewed entry, so the load refuses.
	duplicate := `{"version":2,"staged":[],"staged":[{"package":"p","name":"N","reason":"r","retained":{"classification":"test-support","evidence":"` + stagedSurfaceTestEvidence + `"}}]}`
	if err := os.WriteFile(path, []byte(duplicate), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadStagedSurface(path); err == nil {
		t.Fatal("duplicate staged-surface name accepted")
	}
	duplicateEntry := `{"version":2,"staged":[{"package":"p","name":"N","reason":"r","retained":{"classification":"test-support","evidence":"` + stagedSurfaceTestEvidence + `"}},{"package":"p","name":"N","reason":"r2","retained":{"classification":"test-support","evidence":"` + stagedSurfaceTestEvidence + `"}}]}`
	if err := os.WriteFile(path, []byte(duplicateEntry), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadStagedSurface(path); err == nil {
		t.Fatal("duplicate package+name staged entries accepted")
	}
}

func TestStagedSurfaceResolvesOnlyCanonicalPlanSteps(t *testing.T) {
	if _, err := LoadStagedSurface(filepath.Join("..", "..", "docs", "staged_surface.json")); err != nil {
		t.Fatalf("repository staged surface: %v", err)
	}

	root := t.TempDir()
	planBody := `{"campaign":"fixture","doctrine":"fixture","items":[{"id":"future","title":"Future","status":"open","steps":[{"id":"consume","title":"Consume","status":"open","verify":"go test ./..."},{"id":"blocked","title":"Blocked","status":"blocked: fixture","verify":"go test ./..."}]}]}`
	if err := os.WriteFile(filepath.Join(root, "plan.json"), []byte(planBody), 0o644); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "staged_surface.json")
	manifest := func(reference string) string {
		return fmt.Sprintf(`{"version":2,"staged":[{"package":"p","name":"N","reason":"fixture","retire_with":%q}]}`, reference)
	}
	if err := os.WriteFile(path, []byte(manifest("future/consume")), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadStagedSurface(path); err != nil {
		t.Fatalf("canonical open reference refused: %v", err)
	}
	for _, reference := range []string{"future", "/consume", "future/", "future/consume/extra", "future/missing", "historical/completed", "future/blocked"} {
		t.Run(reference, func(t *testing.T) {
			if err := os.WriteFile(path, []byte(manifest(reference)), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadStagedSurface(path); err == nil {
				t.Fatalf("non-open or historical retire_with %q accepted", reference)
			}
		})
	}
}

func TestStagedSurfaceRetainedClassification(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "staged_surface.json")
	manifest := func(retained string) string {
		return `{"version":2,"staged":[{"package":"p","name":"N","reason":"fixture",` + retained + `}]}`
	}
	for _, classification := range []stagedSurfaceRetentionClass{stagedSurfaceTestSupport, stagedSurfaceProductionInterface} {
		t.Run(string(classification), func(t *testing.T) {
			retained := fmt.Sprintf(`"retained":{"classification":%q,"evidence":%q}`, classification, stagedSurfaceTestEvidence)
			if err := os.WriteFile(path, []byte(manifest(retained)), 0o644); err != nil {
				t.Fatal(err)
			}
			declaration, err := LoadStagedSurface(path)
			if err != nil || declaration.Staged[0].Retained.Classification != classification {
				t.Fatalf("retained classification = %+v, %v", declaration, err)
			}
		})
	}
	notEvidence, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("not evidence"))
	if err != nil {
		t.Fatal(err)
	}
	invalid := map[string]string{
		"unknown":          manifest(`"retained":{"classification":"temporary","evidence":"` + stagedSurfaceTestEvidence + `"}`),
		"missing evidence": manifest(`"retained":{"classification":"test-support","evidence":""}`),
		"wrong evidence":   manifest(`"retained":{"classification":"test-support","evidence":"` + notEvidence.String() + `"}`),
		"neither":          `{"version":2,"staged":[{"package":"p","name":"N","reason":"fixture"}]}`,
		"both":             manifest(`"retire_with":"future/consume","retained":{"classification":"test-support","evidence":"` + stagedSurfaceTestEvidence + `"}`),
		"free text":        `{"version":2,"staged":[{"package":"p","name":"N","reason":"fixture","consumer_trigger":"later"}]}`,
	}
	for name, body := range invalid {
		t.Run(name, func(t *testing.T) {
			if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadStagedSurface(path); err == nil {
				t.Fatal("invalid retained classification accepted")
			}
		})
	}
}
