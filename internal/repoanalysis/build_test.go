package repoanalysis

import (
	"os"
	"path/filepath"
	"testing"
)

func TestProductionConsumerCensus(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "p", "p.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package p\nfunc F() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	selection, err := HostBuildSelection(root, "./...")
	if err != nil {
		t.Fatal(err)
	}
	if !selection.Files["p/p.go"] || selection.Packages["p/p.go"] != "example/p" || selection.Context == "" {
		t.Fatalf("selection = %+v", selection)
	}
}
