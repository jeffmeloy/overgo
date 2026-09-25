package inferencesurface

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestMovesSeesDeletedSurfaceInput holds Moves to the inputs a change
// removes, which the tree it reads no longer lists: a file a closure package
// embeds through a directory or glob pattern can be deleted without editing
// any source, and must still be a move, while a deleted file no pattern
// reaches is not. Deleting a closure package takes an edit to its importer,
// which is a move.
func TestMovesSeesDeletedSurfaceInput(t *testing.T) {
	t.Parallel()
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
	write("internal/inference/inference.go", "package inference\n\nimport \"embed\"\n\n//go:embed catalog rules/*.json\nvar inputs embed.FS\n")
	write("internal/inference/catalog/kept.json", "{}")
	write("internal/inference/rules/kept.json", "{}")
	write("internal/cuda/executor/executor.go", "package executor\n")
	write("internal/modelrecipe/recipe.go", "package modelrecipe\n")
	write("kernels/manifest.json", "{}")
	write("kernels/cuda/ops.cu", "")
	deleted := []string{
		"internal/inference/catalog/nested/removed.json",
		"internal/inference/rules/removed.json",
		"internal/inference/rules/removed.txt",
		"internal/inference/testdata/removed.json",
		"internal/tables/tables.go",
		"docs/removed.md",
	}
	moves, err := Moves(t.Context(), root, append(deleted, "internal/inference/inference.go"))
	want := []string{
		"internal/inference/catalog/nested/removed.json",
		"internal/inference/inference.go",
		"internal/inference/rules/removed.json",
	}
	if err != nil || !slices.Equal(moves, want) {
		t.Fatalf("moves = %v, %v\nwant %v", moves, err, want)
	}
}
