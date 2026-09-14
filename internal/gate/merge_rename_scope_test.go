package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestMergeScopeIncludesBothRenamePaths(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	runGitFixture(t, repo, "config", "diff.renames", "true")
	if err := os.WriteFile(filepath.Join(repo, "before.go"), []byte("package fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "before.go")
	runGitFixture(t, repo, "commit", "-q", "-m", "base")
	runGitFixture(t, repo, "mv", "before.go", "after.go")
	paths, err := stagedPaths(repo)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(paths, []string{"after.go", "before.go"}) {
		t.Fatalf("rename scope = %q; both deletion and addition must ship", paths)
	}
	gate := gateContext{repo: repo, paths: paths}
	if _, err := gate.stepScope(); err != nil {
		t.Fatalf("complete rename scope refused: %v", err)
	}
	gate.paths = []string{"after.go"}
	if _, err := gate.stepScope(); err == nil {
		t.Fatal("scope accepted an omitted rename source")
	}
}
