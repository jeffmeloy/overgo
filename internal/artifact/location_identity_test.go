package artifact

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestCanonicalLocalLocationContract(t *testing.T) {
	root := t.TempDir()
	file := filepath.Join(root, "dataset.jsonl")
	testcheck.NoError(t, os.WriteFile(file, []byte("{}\n"), 0o600))
	id := identifyLocationFixture(t, KindDataset, file)
	first, err := CanonicalLocalLocation(id, LocationFile, file)
	testcheck.NoError(t, err)
	second, err := CanonicalLocalLocation(id, LocationFile, filepath.Join(root, ".", "dataset.jsonl"))
	testcheck.NoError(t, err)
	if first.Value != second.Value || !filepath.IsAbs(first.Value) {
		t.Fatalf("canonical locations = (%+v, %+v)", first, second)
	}
	if first.Artifact != id || first.Value == id.String() {
		t.Fatalf("physical location replaced content identity: %+v", first)
	}
	if _, err := CanonicalLocalLocation(id, LocationDirectory, file); err == nil {
		t.Fatal("file admitted as a directory location")
	}
	testcheck.NoError(t, os.Remove(file))
	if _, err := CanonicalLocalLocation(id, LocationFile, file); err == nil {
		t.Fatal("missing file admitted as a live location")
	}
}

func identifyLocationFixture(t *testing.T, kind Kind, filename string) ID {
	t.Helper()
	file, err := os.Open(filename)
	testcheck.NoError(t, err)
	defer file.Close()
	id, _, err := Identify(kind, file)
	testcheck.NoError(t, err)
	return id
}
