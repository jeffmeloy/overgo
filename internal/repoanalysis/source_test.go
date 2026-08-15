package repoanalysis

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSourceSnapshotParseReuse(t *testing.T) {
	root := t.TempDir()
	content := []byte("package p\nfunc same(v int) int { return v + 1 }\n")
	for _, relative := range []string{"a/a.go", "b/b.go"} {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, content, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := DiscoverGo(root, "a", "b")
	if err != nil {
		t.Fatal(err)
	}
	first, _ := snapshot.Files[0].Syntax()
	second, _ := snapshot.Files[1].Syntax()
	if len(snapshot.Files) != 2 || first != second || snapshot.Files[0].ContentID != snapshot.Files[1].ContentID {
		t.Fatal("identical source content was not reused")
	}
}
