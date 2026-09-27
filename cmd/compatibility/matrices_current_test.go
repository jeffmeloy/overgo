package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestPublishedMatricesAreCurrent holds the committed compatibility and
// training matrices to the claims they are generated from, as
// `compatibility -check` does, so go test answers for them.
func TestPublishedMatricesAreCurrent(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	for path, generator := range map[string]func(string) ([]byte, error){
		matrixPath:         generate,
		trainingMatrixPath: generateTraining,
	} {
		want, err := generator(root)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		got, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s is stale; regenerate with: go run ./cmd/compatibility -update", path)
		}
	}
}
