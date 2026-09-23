package repoanalysis

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gosource"
)

// TestStandardNamesResolveOnce holds package-name resolution to asking go
// list for a standard-library package once per process: the toolchain fixes
// its name, and every code-manifest generation asked again for the same few
// hundred packages. With go off the path the second resolution still names
// them, so it asked nothing.
func TestStandardNamesResolveOnce(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "p", "p.go")
	if err := os.MkdirAll(filepath.Dir(source), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("package p\n\nimport (\n\t\"encoding/json\"\n\t\"net/http/httptest\"\n)\n\nvar _ = json.Valid\nvar _ = httptest.NewRecorder\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := LoadGo(root, []string{"p/p.go"})
	if err != nil {
		t.Fatal(err)
	}
	selection := gosource.BuildSelection{Root: root}
	first, err := PackageNames(snapshot, selection)
	if err != nil || first["encoding/json"] != "json" || first["net/http/httptest"] != "httptest" {
		t.Fatalf("first resolution = %v, %v", first, err)
	}
	t.Setenv("PATH", "")
	second, err := PackageNames(snapshot, selection)
	if err != nil || second["encoding/json"] != "json" || second["net/http/httptest"] != "httptest" {
		t.Fatalf("standard names were asked for again: %v, %v", second, err)
	}
}
