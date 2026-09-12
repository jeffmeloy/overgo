package gate

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestChangedPackageFailurePreventsBroadSweep(t *testing.T) {
	root, _ := packageIdentityFixture(t)
	marker := filepath.Join(t.TempDir(), "consumer-ran")
	files := map[string]string{
		".gitignore":      "tmp/\novergodb-store/\n",
		"NOTES.md":        "changed documentation\n",
		"app/app_test.go": "package app\nimport \"testing\"\nfunc TestRequired(t *testing.T) { t.Fatal(\"changed owner failed\") }\n",
		// The consumer names the changed document, so it is required; a
		// reader that names nothing outside its package is confined.
		"other/other_test.go": fmt.Sprintf("package other\nimport (\"os\"; \"testing\")\nfunc TestRequired(t *testing.T) { if _, err := os.ReadFile(\"../NOTES.md\"); err != nil { t.Fatal(err) }; if err := os.WriteFile(%q, []byte(\"ran\"), 0600); err != nil { t.Fatal(err) } }\n", marker),
	}
	for name, content := range files {
		path := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGitFixture(t, root, "init", "-q")
	environment, err := discoverEnvironment(root)
	if err != nil {
		t.Fatal(err)
	}
	newGate := func() *gateContext {
		snapshot, err := repoanalysis.DiscoverGo(root, "app", "dep", "other")
		if err != nil {
			t.Fatal(err)
		}
		g := &gateContext{repo: root, storePath: StorePath, environment: environment,
			paths: []string{"app/app_test.go", "NOTES.md"}, source: &snapshot}
		t.Cleanup(func() { _ = g.closeStore() })
		return g
	}
	g := newGate()
	if _, err := g.stepTest(t.Context()); err == nil || !strings.Contains(err.Error(), "changed owner failed") {
		t.Fatalf("changed-package failure lost: %v", err)
	}
	if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("broad consumer ran before changed-package acceptance: %v", err)
	}
	if !slices.Contains(g.audit, "package test evidence: 0 reused + 1 executed") {
		t.Fatalf("unstarted packages counted as executed: %v", g.audit)
	}
	if err := os.WriteFile(filepath.Join(root, "app/app_test.go"), []byte("package app\nimport \"testing\"\nfunc TestRequired(t *testing.T) {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := newGate().stepTest(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("successful owner did not release the remaining required checks: %v", err)
	}
}
