package gate

import (
	"os"
	"path/filepath"
	"testing"
)

// TestGateRefusesStrayExecutables pins the commit-time hygiene owner:
// a .exe at the repository root is always an accident -- a
// single-package go build dropping its binary at the working
// directory -- and the gate surfaces it instead of leaving it hidden
// behind the gitignore until the release check.
func TestGateRefusesStrayExecutables(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "bin", "gate.exe"), []byte("binary"), 0o600); err != nil {
		t.Fatal(err)
	}
	stray, err := strayRootExecutables(root)
	if err != nil || len(stray) != 0 {
		t.Fatalf("clean root = (%v, %v)", stray, err)
	}
	if err := os.WriteFile(filepath.Join(root, "loop.exe"), []byte("stray"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "Plan.EXE"), []byte("stray"), 0o600); err != nil {
		t.Fatal(err)
	}
	stray, err = strayRootExecutables(root)
	if err != nil || len(stray) != 2 {
		t.Fatalf("stray root = (%v, %v)", stray, err)
	}
}
