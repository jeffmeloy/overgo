package inferencesurface

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// The file digest is a function of root-relative paths and contents:
// the same files under another root digest the same, one changed byte
// changes it, and the order the files are named in does not matter.
func TestDigestFilesIsPathAndContentBound(t *testing.T) {
	write := func(root string, contents map[string]string) []string {
		var files []string
		for name, content := range contents {
			path := filepath.Join(root, name)
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
			files = append(files, path)
		}
		return files
	}
	contents := map[string]string{"a/x.go": "package a\n", "kernels/k.cu": "__global__ void k() {}\n"}
	first, second := t.TempDir(), t.TempDir()
	firstFiles, secondFiles := write(first, contents), write(second, contents)
	firstDigest, err := digestFiles(first, firstFiles)
	if err != nil {
		t.Fatal(err)
	}
	secondDigest, err := digestFiles(second, []string{secondFiles[1], secondFiles[0]})
	if err != nil {
		t.Fatal(err)
	}
	if firstDigest != secondDigest || len(firstDigest) != 64 {
		t.Fatalf("digests differ across roots: %s vs %s", firstDigest, secondDigest)
	}
	if err := os.WriteFile(filepath.Join(first, "a/x.go"), []byte("package a // changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	changed, err := digestFiles(first, firstFiles)
	if err != nil {
		t.Fatal(err)
	}
	if changed == firstDigest {
		t.Fatal("a changed file left the digest unchanged")
	}
}

// The repository's own inference surface resolves through the Go tool
// and is stable within a process.
func TestSurfaceOfTheRepositoryIsStable(t *testing.T) {
	ctx := t.Context()
	first, err := Digest(ctx, "../..")
	if err != nil {
		t.Fatal(err)
	}
	second, err := Digest(ctx, "../..")
	if err != nil || first != second || len(first) != 64 {
		t.Fatalf("surface = %q, %q, %v", first, second, err)
	}
}

// TestTrainingLeavesTheSurface holds training outside the inference
// surface, so a trainer edit does not expire every text model's guards. The
// surface reached these packages only through composition's graft training
// and its Tier-0 chain probe, which trainingworkflow now owns.
func TestTrainingLeavesTheSurface(t *testing.T) {
	packages, err := Packages(t.Context(), "../..")
	if err != nil {
		t.Fatal(err)
	}
	for _, pkg := range packages {
		if slices.Contains([]string{"overgo/internal/densecausal", "overgo/internal/trainingprogram", "overgo/internal/trainingdata", "overgo/internal/trainingworkflow", "overgo/internal/capabilityruntime"}, pkg.ImportPath) {
			t.Errorf("the inference surface reaches training package %s", pkg.ImportPath)
		}
	}
}

// TestCandidateRealizationLeavesTheSurface holds the adapter trainer and its
// optimizers outside the inference surface. The surface reached them only
// through composition's candidate realization and the improvement driver
// that sequences it, which trainingworkflow now owns.
func TestCandidateRealizationLeavesTheSurface(t *testing.T) {
	packages, err := Packages(t.Context(), "../..")
	if err != nil {
		t.Fatal(err)
	}
	for _, pkg := range packages {
		if slices.Contains([]string{"overgo/internal/bridgetrain", "overgo/internal/hostoptimizer", "overgo/internal/optimizer"}, pkg.ImportPath) {
			t.Errorf("the inference surface reaches training package %s", pkg.ImportPath)
		}
	}
}

// TestDatasetLeavesTheSurface holds the dataset package outside the inference
// surface, so a dataset edit does not expire every text model's guards. The
// surface reached it only through the model recipe's dataset-transform
// admission adapter, which evaluation's knowledge candidates now own.
func TestDatasetLeavesTheSurface(t *testing.T) {
	packages, err := Packages(t.Context(), "../..")
	if err != nil {
		t.Fatal(err)
	}
	for _, pkg := range packages {
		if pkg.ImportPath == "overgo/internal/dataset" {
			t.Errorf("the inference surface reaches dataset package %s", pkg.ImportPath)
		}
	}
}

// TestSurfaceCoversEmbeddedInputsAndModules holds the surface to every
// input inference reads, not only its Go sources: a file a closure package
// embeds (the architecture catalog, the runtime policies, the compiled
// kernels) moves the digest and is a move, go.mod is a move, the version of
// another module the closure compiles is digested, and the repository's own
// closure names its tokenizer's regexp module by version.
func TestSurfaceCoversEmbeddedInputsAndModules(t *testing.T) {
	root := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module fixture\n\ngo 1.26\n")
	write("internal/inference/inference.go", "package inference\n\nimport _ \"embed\"\n\n//go:embed profile.json\nvar profile []byte\n")
	write("internal/inference/profile.json", `{"heads":2}`)
	write("internal/cuda/executor/executor.go", "package executor\n")
	write("internal/modelrecipe/recipe.go", "package modelrecipe\n")
	write("kernels/manifest.json", "{}")
	write("kernels/cuda/ops.cu", "")
	ctx := t.Context()
	before, err := computeSurface(ctx, root)
	if err != nil {
		t.Fatal(err)
	}
	write("internal/inference/profile.json", `{"heads":4}`)
	after, err := computeSurface(ctx, root)
	if err != nil || after == before {
		t.Fatalf("an embedded input change left the surface at %s: %v", before, err)
	}
	moves, err := Moves(ctx, root, []string{"docs/notes.md", "internal/inference/profile.json", "go.mod"})
	if err != nil || !slices.Equal(moves, []string{"go.mod", "internal/inference/profile.json"}) {
		t.Fatalf("moves = %v, %v", moves, err)
	}
	older, olderErr := digestFiles(root, nil, "example.com/tokenizer@v1.0.0")
	newer, newerErr := digestFiles(root, nil, "example.com/tokenizer@v1.0.1")
	if olderErr != nil || newerErr != nil || older == newer {
		t.Fatalf("a module version change left the digest at %s: %v %v", older, olderErr, newerErr)
	}
	_, modules, err := closure(ctx, "../..")
	if err != nil || !slices.ContainsFunc(modules, func(module string) bool { return strings.HasPrefix(module, "github.com/dlclark/regexp2/v2@v") }) {
		t.Fatalf("closure modules = %v, %v", modules, err)
	}
}
