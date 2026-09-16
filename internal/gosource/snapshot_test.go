package gosource

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestSnapshotProvenance(t *testing.T) {
	root := t.TempDir()
	const source = "package fixture\n"
	for _, name := range []string{"a.go", "b.go"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
	}
	paths := []string{"b.go", "./a.go", "a.go", "missing.go", "notes.md"}
	snapshot, err := LoadGo(root, paths)
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Files) != 2 || snapshot.Files[0].Path != "a.go" || snapshot.Files[1].Path != "b.go" || paths[0] != "b.go" {
		t.Fatalf("source membership/order changed: %+v", snapshot)
	}
	// Independent oracle for the existing provenance byte grammar.
	content := sha256.Sum256([]byte(source))
	digest := hex.EncodeToString(content[:])
	want := sha256.Sum256([]byte("a.go\x00" + digest + "\x00b.go\x00" + digest + "\x00"))
	if snapshot.Identity() != hex.EncodeToString(want[:]) {
		t.Fatal("source identity grammar changed")
	}
	if &snapshot.Files[0].Data[0] != &snapshot.Files[1].Data[0] {
		t.Fatal("identical bytes acquired twice")
	}
	if err := os.Rename(filepath.Join(root, "b.go"), filepath.Join(root, "c.go")); err != nil {
		t.Fatal(err)
	}
	for _, paths := range [][]string{{"a.go", "b.go"}, {"a.go", "c.go"}} {
		changed, err := LoadGo(root, paths)
		if err != nil || changed.Identity() == snapshot.Identity() {
			t.Fatalf("removed or renamed source retained identity: %v", err)
		}
	}
	for _, path := range []string{"../escape.go", filepath.Join(root, "a.go")} {
		if _, err := LoadGo(root, []string{path}); err == nil {
			t.Fatalf("accepted escaping path %q", path)
		}
	}
}
