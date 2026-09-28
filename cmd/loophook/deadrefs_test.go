package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestDeadReferencesAreReported holds the session-start scan to what prose
// names: a repository path or a test the tree no longer holds is reported
// once, one that exists is not, and a prefix, a template, a number and a
// one-letter placeholder are not references.
func TestDeadReferencesAreReported(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	for name, content := range map[string]string{
		"cmd/tool/main.go":           "package main\n",
		"internal/present/x.go":      "package present\n",
		"internal/present/x_test.go": "package present\n\nimport \"testing\"\n\nfunc TestAlive(t *testing.T) {}\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	text := "Run cmd/tool, not cmd/gone. Read internal/present/x.go, not internal/absent/y.go (internal/absent/y.go twice). " +
		"TestAlive holds it; TestGone did. The TestAli* prefix, cmd/<name>, docs/29.6 MB and cmd/x are no references."
	got := deadReferences(root, map[string]string{"note": text})
	if want := []string{"note: TestGone", "note: cmd/gone", "note: internal/absent/y.go"}; !slices.Equal(got, want) {
		t.Fatalf("dead references = %q, want %q", got, want)
	}
}
