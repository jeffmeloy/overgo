package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
)

func TestASTStructuralProfileGate(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "p", "p.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package p\nfunc F(v int) int { return v }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"init"}, {"add", "internal/p/p.go"}, {"-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-m", "base"},
	} {
		if _, err := command(root, "git", args...); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(path, []byte("package p\nfunc F(v int) int { if v > 0 { return v }; return 0 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g := gateContext{repo: root, paths: []string{"internal/p/p.go"}}
	skipped, err := g.stepProfile()
	if err != nil || skipped || len(g.honesty) < 4 || !strings.Contains(g.honesty[0], "production=1 files") ||
		!strings.Contains(g.honesty[1], "delta vs HEAD") {
		t.Fatalf("profile step = skipped %v, err %v, honesty %v", skipped, err, g.honesty)
	}
}

func TestGateRejectsNewUnconsumedProductionSurface(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "p", "p.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package p\nfunc Existing() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"init"}, {"add", "."}, {"-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-m", "base"},
	} {
		if _, err := command(root, "git", args...); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(path, []byte("package p\nfunc Existing() {}\nfunc AddedButUnused() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g := gateContext{repo: root, paths: []string{"internal/p/p.go"}}
	if _, err := g.stepProfile(); err == nil || !strings.Contains(err.Error(), "AddedButUnused=zero") {
		t.Fatalf("unconsumed surface error = %v", err)
	}
}

func TestAdvisoryCandidate(t *testing.T) {
	profile := codeprofile.Profile{
		Functions: []codeprofile.Function{
			{File: "internal/other/large.go", Name: "larger", Nodes: 40, Branches: 8},
			{File: "internal/p/p.go", Name: "changed", Nodes: 20, Branches: 3},
			{File: "internal/p/p.go", Name: "validateShape", Nodes: 18, Branches: 4, AdvisoryClass: "validator"},
			{File: "internal/p/p_test.go", Name: "TestChanged", Nodes: 16, Branches: 2, AdvisoryClass: "test"},
		},
		Clones: []codeprofile.Clone{
			{Nodes: 12, Functions: []string{"internal/p/p.go:changed", "internal/q/q.go:peer"}},
			{Nodes: 10, Functions: []string{"internal/p/p.go:validateShape", "internal/q/q.go:validatePeer"}, AdvisoryClass: "validator"},
			{Nodes: 8, Functions: []string{"internal/p/p_test.go:TestChanged", "internal/q/q_test.go:TestPeer"}, AdvisoryClass: "test"},
		},
	}
	got := profileReviewFocus(profile, []string{"internal/p/p.go", "internal/p/p_test.go"})
	for _, want := range []string{
		"production=internal/p/p.go:changed", "validator=internal/p/p.go:validateShape", "test=internal/p/p_test.go:TestChanged",
		"exact_clone_production=nodes=12", "exact_clone_validator=nodes=10", "exact_clone_test=nodes=8",
		"advisory_only=inspect semantic ownership and numerical contracts", "require parity evidence",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("review candidates %q lack %q", got, want)
		}
	}
	if got := profileReviewFocus(profile, []string{"internal/new/empty.go"}); !strings.Contains(got, "production=none") ||
		!strings.Contains(got, "exact_clone_test=none") {
		t.Fatalf("empty review focus = %q", got)
	}
}

func TestSurfaceDeltaHonesty(t *testing.T) {
	base := codeprofile.Profile{
		Production: codeprofile.Partition{Files: 2, Nodes: 100}, Test: codeprofile.Partition{Files: 1, Nodes: 30},
		Functions: []codeprofile.Function{{File: "v.go", Name: "validateBase", Nodes: 20, AdvisoryClass: "validator"}},
		Clones:    []codeprofile.Clone{{Nodes: 10, Functions: []string{"a:f", "b:g"}}}, DuplicateExcessNodes: 10,
	}
	candidate := codeprofile.Profile{
		Production: codeprofile.Partition{Files: 3, Nodes: 125}, Test: codeprofile.Partition{Files: 1, Nodes: 35},
		Functions: []codeprofile.Function{
			{File: "v.go", Name: "validateBase", Nodes: 20, AdvisoryClass: "validator"},
			{File: "v.go", Name: "validateAdded", Nodes: 5, AdvisoryClass: "validator"},
		},
		Clones: []codeprofile.Clone{{Nodes: 6, Functions: []string{"a:f", "b:g"}}}, DuplicateExcessNodes: 6,
		ExportedDeclarations: 2, PackageImportEdges: 1,
	}
	got := surfaceDeltaHonesty(base, candidate)
	for _, want := range []string{
		"production=+1 files/+25 nodes", "test=+0/+5", "validator_subset=+1 functions/+5 nodes",
		"duplicate_excess=-4", "exported=+2", "imports=+1",
		"duplication fell while production grew; reduction does not offset surface growth",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("surface delta %q lacks %q", got, want)
		}
	}
}

func TestAutomationROIProjection(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "p", "p.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package p\nfunc keep() {}\nfunc remove() { println(1) }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	base, err := repoanalysis.LoadGo(root, []string{"internal/p/p.go"})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := base.Overlay(map[string][]byte{
		"internal/p/p.go": []byte("package p\nfunc keep() {}\n"),
		"internal/q/q.go": []byte("package q\nfunc added(v int) int { if v > 0 { return v }; return 0 }\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	movement, err := codeprofile.MeasureProductionMovement(base, candidate)
	if err != nil {
		t.Fatal(err)
	}
	got, err := automationROIAdmission("plan-slice@fixture", movement)
	if movement.Added == 0 || movement.Deleted == 0 || movement.GoLinesAdded == 0 || movement.GoLinesDeleted == 0 ||
		!strings.Contains(got, "production_ast=") || !strings.Contains(got, "go_lines=") || err == nil {
		t.Fatalf("automation ROI = %+v, %q, %v", movement, got, err)
	}
	if compact := compactHonesty([]string{got}); len(compact) != 1 || !strings.HasPrefix(compact[0], "roi: ") {
		t.Fatalf("automation ROI hidden from gate summary: %v", compact)
	}
	if _, err := automationROIAdmission("deletion", codeprofile.ProductionMovement{Deleted: 1, GoLinesDeleted: 1}); err != nil {
		t.Fatalf("deletion-only wave rejected: %v", err)
	}
}

func TestASTProfileEvidenceGate(t *testing.T) {
	gate, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("gate"))
	if err != nil {
		t.Fatal(err)
	}
	profile := codeProfileFixture()
	g := gateContext{profile: &profile}
	var batch artifact.Batch
	if err := g.appendProfileEvidence(&batch, "0123456789abcdef0123456789abcdef01234567", gate); err != nil {
		t.Fatal(err)
	}
	if len(batch.Contents) != 1 || len(batch.Lineage) != 1 {
		t.Fatalf("profile batch = %+v", batch)
	}
	g.profileDirty = true
	batch = artifact.Batch{}
	if err := g.appendProfileEvidence(&batch, "0123456789abcdef0123456789abcdef01234567", gate); err != nil {
		t.Fatal(err)
	}
	if len(batch.Contents) != 0 || !strings.Contains(g.honesty[len(g.honesty)-1], "not persisted") {
		t.Fatalf("contaminated profile was persisted: %+v", batch)
	}
}

func codeProfileFixture() codeprofile.Profile {
	return codeprofile.Profile{Production: codeprofile.Partition{Files: 1, Nodes: 1}}
}
