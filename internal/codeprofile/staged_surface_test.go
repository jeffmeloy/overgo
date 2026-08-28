package codeprofile

import (
	"os"
	"path/filepath"
	"testing"
)

// TestStagedSurfacePartition pins the acceptance contract: a declared
// package+name pair is accepted, everything else keeps blocking, an
// absent file stages nothing, and an incomplete or duplicate-name entry
// refuses to load.
func TestStagedSurfacePartition(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "staged_surface.json")
	declaration, err := LoadStagedSurface(path)
	if err != nil || len(declaration.Staged) != 0 {
		t.Fatalf("absent declaration = %+v, %v", declaration, err)
	}
	body := `{"version":1,"staged":[{"package":"overgo/internal/agentloop","name":"NewDelegatedCoordinator","reason":"agent orchestration API","consumer_trigger":"agent delegation campaign"}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	declaration, err = LoadStagedSurface(path)
	if err != nil || len(declaration.Staged) != 1 {
		t.Fatalf("declaration = %+v, %v", declaration, err)
	}
	unconsumed := []ConsumerDeclaration{
		{Package: "overgo/internal/agentloop", Name: "NewDelegatedCoordinator", Kind: "func"},
		{Package: "overgo/internal/agentloop", Name: "Undeclared", Kind: "func"},
	}
	accepted, blocking := PartitionStagedSurface(unconsumed, declaration)
	if len(accepted) != 1 || accepted[0].Name != "NewDelegatedCoordinator" ||
		len(blocking) != 1 || blocking[0].Name != "Undeclared" {
		t.Fatalf("partition = accepted %+v blocking %+v", accepted, blocking)
	}
	incomplete := `{"version":1,"staged":[{"package":"p","name":"N","reason":"","consumer_trigger":"t"}]}`
	if err := os.WriteFile(path, []byte(incomplete), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadStagedSurface(path); err == nil {
		t.Fatal("incomplete staged entry accepted")
	}
	// Gate authority decodes strictly: a duplicate name would silently
	// last-wins and drop a reviewed entry, so the load refuses.
	duplicate := `{"version":1,"staged":[],"staged":[{"package":"p","name":"N","reason":"r","consumer_trigger":"t"}]}`
	if err := os.WriteFile(path, []byte(duplicate), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadStagedSurface(path); err == nil {
		t.Fatal("duplicate staged-surface name accepted")
	}
}
