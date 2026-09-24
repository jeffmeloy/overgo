package gitauthority

import (
	"os"
	"path/filepath"
	"testing"
)

// TestAncestorFilesAcceptsPendingMergeParent reads a revision that only the
// incoming side of a pending merge descends from: the merge commit will
// record it as a parent, so a lane merging master can inherit what master's
// history holds. Once the merge is abandoned the revision is refused again.
func TestAncestorFilesAcceptsPendingMergeParent(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	gitTestCommand(t, root, "init", "-q", "--initial-branch=main")
	gitTestCommand(t, root, "config", "user.name", "fixture")
	gitTestCommand(t, root, "config", "user.email", "fixture@example.invalid")
	commit := func(name, value string) string {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, name), []byte(value), 0o644); err != nil {
			t.Fatal(err)
		}
		gitTestCommand(t, root, "add", name)
		gitTestCommand(t, root, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
		return gitTestCommand(t, root, "rev-parse", "HEAD")
	}
	commit("contract.json", "base")
	gitTestCommand(t, root, "checkout", "-q", "-b", "incoming")
	incoming := commit("contract.json", "incoming")
	gitTestCommand(t, root, "checkout", "-q", "main")
	commit("lane.json", "lane")
	if _, err := AncestorFiles(t.Context(), root, incoming, "contract.json"); err == nil {
		t.Fatal("accepted the incoming side before any merge")
	}
	gitTestCommand(t, root, "merge", "--no-ff", "--no-commit", "-q", "incoming")
	files, err := AncestorFiles(t.Context(), root, incoming, "contract.json")
	if err != nil || len(files) != 1 || string(files[0]) != "incoming" {
		t.Fatalf("pending merge parent read = %q, %v", files, err)
	}
	gitTestCommand(t, root, "merge", "--abort")
	if _, err := AncestorFiles(t.Context(), root, incoming, "contract.json"); err == nil {
		t.Fatal("accepted the incoming side after the merge was abandoned")
	}
}
