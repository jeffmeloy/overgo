package guard

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"overgo/internal/processlock"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheGuardKeepsOtherSessionsOutOfTheLeadsCheckout holds the guard to the checkout's holder: on 2026-11-08 another
// session that had declared nothing wrote a document into the lead's checkout, and every landing stopped on it. While
// the lead holds the checkout, another session, whatever it declared, is refused a write of a file the repository
// tracks or would track there, by a Write or an Edit and by a shell command that redirects into it, appends to it,
// copies onto it or creates it, from inside the checkout and from a worktree of its own. The lead is refused none of
// it; a path the repository ignores, one outside the checkout and a command that writes nothing are any session's; and
// a hold whose session has ended keeps no one out.
func TestTheGuardKeepsOtherSessionsOutOfTheLeadsCheckout(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	git := func(arguments ...string) {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", append([]string{"-C", root, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "-c", "core.hooksPath="}, arguments...)...)
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v %s", arguments, err, output)
		}
	}
	git("init", "-q")
	testutil.WriteTextFile(t, root, ".gitignore", "tmp/\n"+processlock.StateDirectory+"/\n", 0o600)
	testutil.WriteTextFile(t, root, "internal/p/p.go", "package p\n", 0o600)
	git("add", "-A")
	git("commit", "-q", "-m", "fixture")
	beside := filepath.Join(t.TempDir(), "beside")
	git("worktree", "add", "-q", "--detach", beside)
	hold := func(process int) {
		testutil.WriteTextFile(t, root, processlock.StateDirectory+"/"+WorktreeHolder, HoldText("lead", process), 0o600)
	}
	hold(os.Getpid())
	call := func(session, directory, tool, command, path string) string {
		request := ToolCall{ToolName: tool, Session: session, Directory: directory}
		request.ToolInput.Command, request.ToolInput.Path = command, path
		message, _ := Verdict(request, directory)
		return message
	}
	plan := filepath.Join(root, "docs", "plan.md")
	for _, write := range []struct{ name, tool, command, path string }{
		{"a new document", "Write", "", plan},
		{"an edit of a tracked file", "Edit", "", filepath.Join(root, "internal", "p", "p.go")},
		{"a redirection into a new document", "Bash", "cat > docs/plan.md <<'EOF'\n# plan\nEOF", ""},
		{"an append to a tracked file", "Bash", "echo x >> internal/p/p.go", ""},
		{"a copy onto a new file", "Bash", "cp tmp/a.md docs/b.md", ""},
		{"a file tee creates", "Bash", "echo x | tee docs/c.md", ""},
	} {
		if call("other", root, write.tool, write.command, write.path) == "" {
			t.Errorf("another session was allowed %s in the lead's checkout", write.name)
		}
		if message := call("lead", root, write.tool, write.command, write.path); message != "" {
			t.Errorf("the lead was refused %s: %s", write.name, message)
		}
	}
	if call("other", beside, "Write", "", plan) == "" || call("other", beside, "Bash", "echo x > "+filepath.ToSlash(plan), "") == "" {
		t.Error("a session in a worktree of its own was allowed a write into the lead's checkout")
	}
	if message := call("other", beside, "Write", "", filepath.Join(beside, "docs", "plan.md")); message != "" {
		t.Errorf("a session was refused a write into its own worktree: %s", message)
	}
	for _, allowed := range []struct{ name, tool, command, path string }{
		{"a scratch file the repository ignores", "Write", "", filepath.Join(root, "tmp", "probe.md")},
		{"a redirection into scratch", "Bash", "go test ./... > tmp/test.log 2>&1", ""},
		{"a file outside the checkout", "Write", "", filepath.Join(t.TempDir(), "notes.md")},
		{"a command that writes nothing", "Bash", "go vet ./... 2>/dev/null", ""},
	} {
		if message := call("other", root, allowed.tool, allowed.command, allowed.path); message != "" {
			t.Errorf("another session was refused %s: %s", allowed.name, message)
		}
	}
	ended := exec.CommandContext(t.Context(), "git", "--version")
	testcheck.NoError(t, ended.Run())
	hold(ended.Process.Pid)
	if message := call("other", root, "Write", "", plan); message != "" {
		t.Errorf("a hold whose session has ended kept another session out: %s", message)
	}
}
