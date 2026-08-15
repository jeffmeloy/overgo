package closurescan

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestClosureScanSnapshotConsumer(t *testing.T) {
	root := t.TempDir()
	for relative, content := range map[string]string{
		"internal/live.go":      "package p\nconst Limit = 3\n",
		"internal/other.go":     "package p\nconst Other = 4\n",
		"internal/live_test.go": "package p\nconst TestLimit = 5\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	candidates, err := ScanSnapshot(snapshot, []string{"internal/live.go"})
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 1 || candidates[0].Name != "Limit" {
		t.Fatalf("candidates = %+v", candidates)
	}
}
