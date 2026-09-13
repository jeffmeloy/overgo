package repoanalysis

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHostDependencySourceSelection(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	for name, body := range map[string]string{
		"go.mod":                "module example\n\ngo 1.26\n",
		"reader/reader.go":      "package reader\nimport \"strings\"\nfunc Value() string { return strings.TrimSpace(\" one \") }\n",
		"reader/reader_test.go": "package reader\nimport \"testing\"\nfunc TestReader(t *testing.T) {}\n",
		"reader/disabled.go":    "//go:build dependency_source_disabled\n\npackage reader\nfunc Disabled() {}\n",
		"client/client.go":      "package client\nimport \"example/reader\"\nfunc Value() string { return reader.Value() }\n",
		"client/client_test.go": "package client\nimport \"testing\"\nfunc TestClient(t *testing.T) {}\n",
		"foreign/foreign.go":    "package foreign\nfunc Value() {}\n",
	} {
		path := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	selected, err := HostBuildSelection(root, "-deps", "-test", "./client")
	if err != nil {
		t.Fatal(err)
	}
	if len(selected.Files) != 3 || !selected.Files["reader/reader.go"] || !selected.Files["client/client.go"] || !selected.Files["client/client_test.go"] {
		t.Fatalf("dependency selection includes foreign tests, inactive sources or generated drivers: %+v", selected)
	}
	if selected.Packages["client/client_test.go"] != "example/client" {
		t.Fatalf("test variant escaped package ownership: %+v", selected.Packages)
	}
	production, err := HostBuildSelection(root, "-deps", "./client")
	if err != nil || len(production.Files) != 2 || production.Files["client/client_test.go"] {
		t.Fatalf("producer selection: %+v, %v", production, err)
	}
	legacy, err := HostBuildSelection(root, "./reader")
	if err != nil || !legacy.Files["reader/reader_test.go"] {
		t.Fatalf("AST census lost test membership: %+v, %v", legacy, err)
	}
	if err := os.WriteFile(filepath.Join(root, "client/client.go"), []byte("package client\nimport _ \"example/missing\"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := HostBuildSelection(root, "-deps", "./client"); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("unresolved dependency accepted or unexplained: %v", err)
	}
}
