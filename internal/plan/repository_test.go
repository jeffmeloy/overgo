package plan

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

// planRepository seeds a git repository whose committed plan is document,
// with an empty store beside it, and returns its root.
func planRepository(t *testing.T, document Plan) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := Save(filepath.Join(root, filepath.FromSlash(Path)), document); err != nil {
		t.Fatal(err)
	}
	git(t, root, "init", "-q")
	git(t, root, "config", "user.email", "overgo@example.invalid")
	git(t, root, "config", "user.name", "Overgo Test")
	git(t, root, "add", Path)
	git(t, root, "commit", "-q", "-m", "seed plan")
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return root
}

// git runs one git command in repository and returns its trimmed output.
func git(t *testing.T, repository string, arguments ...string) string {
	t.Helper()
	command := exec.Command("git", arguments...)
	command.Dir = repository
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v: %s", strings.Join(arguments, " "), err, strings.TrimSpace(string(output)))
	}
	return strings.TrimSpace(string(output))
}
