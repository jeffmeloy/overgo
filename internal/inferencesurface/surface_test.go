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

// TestSurfaceInvariants pins the packages that must stay out of the
// inference surface, one row per package with the reason no decode runs it:
// inside the surface, an edit to any of them would expire every text model's
// long-form guards. A row states what must never come back into the surface,
// never the surface's current package list, so serving code may join or
// leave it freely.
//
// To keep a package out, add a row, never a test function: its directory
// and, in one line, why a decode never runs it. A row whose package is gone
// fails, so a rename cannot silently empty it: rename the row with the
// package, or delete the row with it.
func TestSurfaceInvariants(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	module, err := modulePath(root)
	if err != nil {
		t.Fatal(err)
	}
	packages, err := Packages(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	surface := map[string]bool{}
	for _, pkg := range packages {
		surface[pkg.ImportPath] = true
	}
	for _, row := range []struct{ directory, reason string }{
		{"internal/trainingprogram", "it compiles training authority for trainers; graft training carried it in until trainingworkflow took that over"},
		{"internal/trainingdata", "it materializes training streams; a decode reads no dataset"},
		{"internal/trainingworkflow", "it runs training workflows, graft training and candidate realization among them"},
		{"internal/densecausal", "it trains dense-causal models on the host; graft training carried it in until trainingworkflow took that over"},
		{"internal/densecausaltrain", "it trains dense-causal models on the device; only the commands that train import it"},
		{"internal/scratchtrain", "it trains scratch models on the device; only trainers import it"},
		{"internal/optimizer", "its device Muon steppers update weights; a decode steps no optimizer"},
		{"internal/hostoptimizer", "it holds Muon groups and optimizer state; a decode steps no optimizer"},
		{"internal/bridgetrain", "it trains adapter bridges; candidate realization carried it in until trainingworkflow took that over"},
		{"internal/dataset", "it owns dataset documents; the recipe's dataset-transform admission carried it in until evaluation's knowledge candidates took that over"},
		{"internal/devicegrad", "it holds the device kernels only trainers call"},
		{"internal/capabilityruntime", "it places and executes capabilities for workflows; graft training carried it in until trainingworkflow took that over"},
		{"internal/overgodb", "the runner reaches the store only to resolve artifacts by content identity, which cannot change what a decode computes"},
		{"internal/runrecord", "the runner reaches run records only to record artifacts by content identity, which cannot change what a decode computes"},
	} {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(row.directory))); err != nil {
			t.Errorf("row %s holds nothing; rename it with its package or delete it: %v", row.directory, err)
			continue
		}
		if surface[module+"/"+row.directory] {
			t.Errorf("the inference surface reaches %s: %s", row.directory, row.reason)
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
