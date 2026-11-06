package plan

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

// planRepository seeds a git repository whose committed plan is document,
// with an empty store beside it, and returns its root.
func planRepository(t *testing.T, document Plan) string {
	t.Helper()
	root := t.TempDir()
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "docs"), 0o755))
	testcheck.NoError(t, Save(filepath.Join(root, filepath.FromSlash(Path)), document))
	git(t, root, "init", "-q")
	git(t, root, "config", "user.email", "overgo@example.invalid")
	git(t, root, "config", "user.name", "Overgo Test")
	git(t, root, "add", Path)
	git(t, root, "commit", "-q", "-m", "seed plan")
	store := testcheck.Must(overgodb.Create(filepath.Join(root, "overgodb-store")))(t)
	testcheck.NoError(t, store.Close())
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
