package repoanalysis

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

func TestDocsHoldNoUnplannedBinaries(t *testing.T) {
	t.Parallel()
	testcheck.NoError(t, ValidateDocsInventory(filepath.Join("..", "..")))
}

func TestDocsInventoryPolicy(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		path    string
		allowed bool
	}{
		{"nested/readme.md", true}, {"plan.json", true}, {"records.jsonl", true},
		{"notes.txt", true}, {"template.jinja", true}, {".loop_state", false},
		{".dispatch", false}, {"assets/nested/view.PNG", true},
		{"media_samples/speech.wav", false}, {"verification/trace.svg", true},
		{"assets/demo.gif", true}, {"gui/screens/view.png", false},
		{"assets-extra/view.png", false}, {"assets/paper.pdf", false},
		{"nested/.loop_state", false}, {"paper.pdf", false},
	} {
		t.Run(test.path, func(t *testing.T) {
			root := t.TempDir()
			testcheck.NoError(t, os.WriteFile(filepath.Join(root, "README.md"), nil, 0o644))
			path := filepath.Join(root, "docs", filepath.FromSlash(test.path))
			testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
			testcheck.NoError(t, os.WriteFile(path, []byte("fixture"), 0o644))
			err := ValidateDocsInventory(root)
			if (err == nil) != test.allowed {
				t.Fatalf("allowed=%v, error=%v", test.allowed, err)
			}
			if err != nil && !strings.Contains(err.Error(), test.path) {
				t.Fatal(err)
			}
		})
	}
	t.Run("missing directory", func(t *testing.T) {
		if err := ValidateDocsInventory(t.TempDir()); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("want missing directory: %v", err)
		}
	})
}

func TestReadmeReferences(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	for _, directory := range []string{"docs", "cmd/live"} {
		testcheck.NoError(t, os.MkdirAll(filepath.Join(root, directory), 0o755))
	}
	for _, test := range []struct {
		text  string
		valid bool
	}{
		{"[docs](docs) [anchor](#local) [external](https://example.com) `cmd/live`", true},
		{"[missing](docs/missing.json)", false},
		{"`cmd/missing`", false},
	} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, "README.md"), []byte(test.text), 0o644))
		if err := ValidateDocsInventory(root); (err == nil) != test.valid {
			t.Fatalf("valid=%v error=%v", test.valid, err)
		}
	}
}
