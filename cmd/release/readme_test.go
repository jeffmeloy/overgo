package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestReadmeReferencesLiveRepositorySurfaces(t *testing.T) {
	if err := repoanalysis.ValidateDocsInventory(filepath.Join("..", "..")); err != nil {
		t.Fatal(err)
	}
}
