package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
)

func TestDeviceImpactUsesSymbolOwnership(t *testing.T) {
	root := t.TempDir()
	graph := packageInputGraph{root: root, nodes: []goPackageInput{
		{ImportPath: "example/internal/cuda/executor", Dir: filepath.Join(root, "internal", "cuda", "executor")},
		{ImportPath: "example/internal/optimizer", Dir: filepath.Join(root, "internal", "optimizer"), Imports: []string{"example/internal/cuda/executor"}},
		{ImportPath: "example/internal/model", Dir: filepath.Join(root, "internal", "model")},
	}}
	packages, err := graph.dependentDirectories("internal/cuda")
	if err != nil || !slices.Equal(packages, []string{"internal/cuda/executor", "internal/optimizer"}) {
		t.Fatalf("device ownership = %v, %v", packages, err)
	}
}

func TestPackageInputIdentityTracksTransitiveFiles(t *testing.T) {
	root, graph := packageIdentityFixture(t)
	first, err := graph.identity("example/app")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "dep", "dep.go"), []byte("package dep\nconst Value = 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	second, err := graph.identity("example/app")
	if err != nil {
		t.Fatal(err)
	}
	if first == second {
		t.Fatal("transitive dependency change retained package input identity")
	}
}

func TestPackageEvidenceCacheUsesExactInput(t *testing.T) {
	root, graph := packageIdentityFixture(t)
	input, err := graph.identity("example/app")
	if err != nil {
		t.Fatal(err)
	}
	environment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte(root))
	cache := automationcheck.NewEvidenceCache(environment)
	if err := cache.RecordPackagePass("example/app", "short", input); err != nil {
		t.Fatal(err)
	}
	if hit, err := cache.PackageReusable("example/app", "short", input); err != nil || !hit {
		t.Fatalf("exact package pass = %t, %v", hit, err)
	}
	if hit, err := cache.PackageReusable("example/app", "complete", input); err != nil || hit {
		t.Fatalf("different mode reused = %t, %v", hit, err)
	}
}

func TestPackageSelectiveReuse(t *testing.T) {
	root, graph := packageIdentityFixture(t)
	graph.nodes = append(graph.nodes, goPackageInput{ImportPath: "example/other", Dir: filepath.Join(root, "other"), GoFiles: []string{"other.go"}})
	graph.byID["example/other"] = []int{len(graph.nodes) - 1}
	if err := os.MkdirAll(filepath.Join(root, "other"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "other", "other.go"), []byte("package other\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	appInput, _ := graph.identity("example/app")
	otherInput, _ := graph.identity("example/other")
	environment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte(root))
	cache := automationcheck.NewEvidenceCache(environment)
	if err := cache.RecordPackagePass("example/app", "short", appInput); err != nil {
		t.Fatal(err)
	}
	if err := cache.RecordPackagePass("example/other", "short", otherInput); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "dep", "dep.go"), []byte("package dep\nconst Value = 3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	changedApp, _ := graph.identity("example/app")
	unchangedOther, _ := graph.identity("example/other")
	if hit, _ := cache.PackageReusable("example/app", "short", changedApp); hit {
		t.Fatal("changed transitive package evidence was reused")
	}
	if hit, _ := cache.PackageReusable("example/other", "short", unchangedOther); !hit {
		t.Fatal("unaffected package evidence was not reused")
	}
}

func packageIdentityFixture(t *testing.T) (string, packageInputGraph) {
	t.Helper()
	root := t.TempDir()
	for _, directory := range []string{"app", "dep"} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	files := map[string]string{
		"go.mod":     "module example\n\ngo 1.25\n",
		"app/app.go": "package app\nimport _ \"example/dep\"\n",
		"dep/dep.go": "package dep\nconst Value = 1\n",
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(name)), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	nodes := []goPackageInput{
		{ImportPath: "example/app", Dir: filepath.Join(root, "app"), Imports: []string{"example/dep"}, GoFiles: []string{"app.go"}},
		{ImportPath: "example/dep", Dir: filepath.Join(root, "dep"), GoFiles: []string{"dep.go"}},
	}
	return root, packageInputGraph{root: root, nodes: nodes, byID: map[string][]int{"example/app": {0}, "example/dep": {1}}}
}
