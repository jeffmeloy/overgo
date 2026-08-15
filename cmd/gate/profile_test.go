package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/codeprofile"
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
	g := gateContext{repo: root, paths: []string{"internal/p/p.go"}}
	skipped, err := g.stepProfile()
	if err != nil || skipped || len(g.honesty) != 2 || !strings.Contains(g.honesty[0], "production=1 files") {
		t.Fatalf("profile step = skipped %v, err %v, honesty %v", skipped, err, g.honesty)
	}
}

func TestASTProfileReviewFocus(t *testing.T) {
	profile := codeprofile.Profile{
		Functions: []codeprofile.Function{
			{File: "internal/other/large.go", Name: "larger", Nodes: 40, Branches: 8},
			{File: "internal/p/p.go", Name: "changed", Nodes: 20, Branches: 3},
		},
		Clones: []codeprofile.Clone{
			{Nodes: 12, Functions: []string{"internal/p/p.go:changed", "internal/q/q.go:peer"}},
		},
	}
	want := "code review focus: largest_function_in_changed_file=internal/p/p.go:changed nodes=20 branches=3; " +
		"largest_clone_touching_changed_file=nodes=12 functions=internal/p/p.go:changed,internal/q/q.go:peer"
	if got := profileReviewFocus(profile, []string{"internal/p/p.go"}); got != want {
		t.Fatalf("review focus = %q, want %q", got, want)
	}
	if got := profileReviewFocus(profile, []string{"internal/new/empty.go"}); !strings.HasSuffix(got, "largest_clone_touching_changed_file=none") ||
		!strings.Contains(got, "largest_function_in_changed_file=none") {
		t.Fatalf("empty review focus = %q", got)
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
