package artifact

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCanonicalLocalLocationContract(t *testing.T) {
	root := t.TempDir()
	file := filepath.Join(root, "dataset.jsonl")
	if err := os.WriteFile(file, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	id := identifyLocationFixture(t, KindDataset, file)
	first, err := CanonicalLocalLocation(id, LocationFile, file)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CanonicalLocalLocation(id, LocationFile, filepath.Join(root, ".", "dataset.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	same, err := SameLocalLocation(first, second)
	if err != nil || !same || first.Value != second.Value || !filepath.IsAbs(first.Value) {
		t.Fatalf("canonical locations = (%+v, %+v, %t, %v)", first, second, same, err)
	}
	otherID := identifyLocationFixture(t, KindFile, file)
	other := second
	other.Artifact = otherID
	if same, err := SameLocalLocation(first, other); err != nil || same {
		t.Fatalf("different artifact location = (%t, %v)", same, err)
	}
	if _, err := CanonicalLocalLocation(id, LocationDirectory, file); err == nil {
		t.Fatal("file admitted as a directory location")
	}
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	if _, err := CanonicalLocalLocation(id, LocationFile, file); err == nil {
		t.Fatal("missing file admitted as a live location")
	}
}

func identifyLocationFixture(t *testing.T, kind Kind, filename string) ID {
	t.Helper()
	file, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	id, _, err := Identify(kind, file)
	if err != nil {
		t.Fatal(err)
	}
	return id
}
