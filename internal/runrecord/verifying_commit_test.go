package runrecord

import (
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

func TestVerifyingCommitRequiresCleanWorktree(t *testing.T) {
	root := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		command := exec.Command("git", append([]string{"-C", root}, args...)...)
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	git("init")
	git("config", "user.name", "Overgo Test")
	git("config", "user.email", "overgo@example.invalid")
	testutil.WriteTextFile(t, root, "source.go", "package source\n")
	git("add", "source.go")
	git("commit", "-m", "fixture")
	want := git("rev-parse", "HEAD")
	if got, err := VerifyingCommit(root); err != nil || got != want {
		t.Fatalf("clean verifying commit = %q, %v; want %q", got, err, want)
	}

	testutil.WriteTextFile(t, root, "source.go", "package changed\n")
	if _, err := VerifyingCommit(root); err == nil || !strings.Contains(err.Error(), "dirty") {
		t.Fatalf("tracked change error = %v", err)
	}
	git("restore", "source.go")
	testutil.WriteTextFile(t, root, "untracked.go", "package source\n")
	if _, err := VerifyingCommit(root); err == nil || !strings.Contains(err.Error(), "dirty") {
		t.Fatalf("untracked source error = %v", err)
	}
}
