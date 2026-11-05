package gosource

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestProductionConsumerCensus(t *testing.T) {
	root := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example\n\ngo 1.26\n"), 0o644))
	path := filepath.Join(root, "p", "p.go")
	testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	testcheck.NoError(t, os.WriteFile(path, []byte("package p\nfunc F() {}\n"), 0o644))
	selection, err := HostBuildSelection(root, "./...")
	testcheck.NoError(t, err)
	if !selection.Files["p/p.go"] || selection.Packages["p/p.go"] != "example/p" || selection.Context == "" {
		t.Fatalf("selection = %+v", selection)
	}
}
