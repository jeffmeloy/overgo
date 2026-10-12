package guard

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/processlock"
)

// TestReviewSessionRemovesNothing holds the guard to who may run the one
// command that removes leftovers. A review session, and a session in a
// worktree another session holds, are refused it, with a confirming token and
// without: neither has anything of its own here to remove. An ordinary
// session, and a harness that names none, are allowed it, since the command
// itself decides what is finished and asks for the token its listing printed.
// The listing alone removes nothing and is any session's to run. The refusal
// of git worktree remove names that command, which follows no link, in place
// of sending the session to the user. And a delete by hand under the data
// root's experiments folder is refused, as one under its models and
// checkpoints is: an experiment's outputs go only through the command.
func TestReviewSessionRemovesNothing(t *testing.T) {
	root := t.TempDir()
	if output, err := exec.CommandContext(t.Context(), "git", "-C", root, "init", "-q").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v %s", err, output)
	}
	state := filepath.Join(root, processlock.StateDirectory)
	if err := os.MkdirAll(state, 0o755); err != nil {
		t.Fatal(err)
	}
	for session, content := range map[string]string{"reviewer": `{"review": true}`, "second": `{"holder": "s9"}`, "implementer": `{"head": "abc123"}`} {
		if err := os.WriteFile(filepath.Join(state, SessionState(session)), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	call := func(session, command string) string {
		request := ToolCall{ToolName: "Bash", Session: session}
		request.ToolInput.Command = command
		message, _ := Verdict(request, root)
		return message
	}
	for _, command := range []string{
		"go run ./cmd/loop -leftovers -delete experiment",
		"go run ./cmd/loop -leftovers -delete experiment -confirm 0123456789abcdef",
		"go run ./cmd/loop -leftovers --delete=helper-copy",
	} {
		for _, refused := range []string{"reviewer", "second"} {
			if message := call(refused, command); !strings.Contains(message, "removes nothing here") {
				t.Errorf("session %q was answered %q for %q, want it refused as one that removes nothing here", refused, message, command)
			}
		}
		for _, allowed := range []string{"implementer", "", "a session with no state"} {
			if message := call(allowed, command); message != "" {
				t.Errorf("session %q was refused %q: %s", allowed, command, message)
			}
		}
	}
	if message := call("reviewer", "go run ./cmd/loop -leftovers"); message != "" {
		t.Errorf("a review session was refused the listing, which removes nothing: %s", message)
	}
	if message := call("implementer", "git worktree remove C:/t/copy"); !strings.Contains(message, "go run ./cmd/loop -leftovers -delete helper-copy") || strings.Contains(message, "ask the user") {
		t.Errorf("git worktree remove is answered with %q, want a refusal that names the command that clears a helper copy and sends nobody to the user", message)
	}
	for _, command := range []string{"rm -r C:/Users/x/overgo_data/experiments/pilot/run", `rm -r C:\data\experiments\pilot`, "mv C:/data/experiments/pilot C:/t/out"} {
		if message := call("implementer", command); !strings.Contains(message, "experiments/") {
			t.Errorf("%q is answered with %q, want a refusal that names the experiments folder as protected", command, message)
		}
	}
	if message := call("implementer", "find C:/data/experiments/ -name '*.bin' -delete"); message == "" {
		t.Error("a find that deletes under the experiments folder was allowed")
	}
	if message := call("implementer", "ls C:/data/experiments/pilot"); message != "" {
		t.Errorf("reading under the experiments folder was refused: %s", message)
	}
}

// TestReviewSessionEditsNothingTracked holds the guard to what a review
// session may not do. A session whose state says it is a review session is
// refused an edit of a file the repository tracks or would track, and a
// landing; it may write a file the repository ignores or one outside the
// checkout, and run anything else. The same calls from a session that is not
// a review session, and from a harness that names none, are allowed: the rule
// is the review session's and no other's.
func TestReviewSessionEditsNothingTracked(t *testing.T) {
	root := t.TempDir()
	if output, err := exec.CommandContext(t.Context(), "git", "-C", root, "init", "-q").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v %s", err, output)
	}
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(".gitignore", "tmp/\n"+processlock.StateDirectory+"/\n")
	write(processlock.StateDirectory+"/"+SessionState("reviewer"), `{"head": "abc123", "facts": [], "review": true}`)
	write(processlock.StateDirectory+"/"+SessionState("implementer"), `{"head": "abc123", "facts": []}`)
	call := func(session, tool, command, path string) string {
		request := ToolCall{ToolName: tool, Session: session}
		request.ToolInput.Command, request.ToolInput.Path = command, path
		message, _ := Verdict(request, root)
		return message
	}
	tracked, ignored, outside := filepath.Join(root, "internal", "p", "p.go"), filepath.Join(root, "tmp", "probe.go"), filepath.Join(t.TempDir(), "notes.md")
	for _, refused := range []struct{ name, tool, command, path string }{
		{"an edit of a file the repository would track", "Edit", "", tracked},
		{"a new file the repository would track", "Write", "", filepath.Join(root, "docs", "review.md")},
		{"a landing", "Bash", "go run ./cmd/loop -land row/do -message-file - < message.txt", ""},
		{"the gate itself", "Bash", "go run ./cmd/gate -plan row/do -message-file -", ""},
	} {
		if message := call("reviewer", refused.tool, refused.command, refused.path); message == "" {
			t.Errorf("a review session was allowed %s", refused.name)
		}
		for _, other := range []string{"implementer", "", "a session with no state"} {
			if message := call(other, refused.tool, refused.command, refused.path); message != "" {
				t.Errorf("session %q was refused %s: %s", other, refused.name, message)
			}
		}
	}
	for _, allowed := range []struct{ name, tool, command, path string }{
		{"a scratch file the repository ignores", "Write", "", ignored},
		{"a file outside the checkout", "Write", "", outside},
		{"a test run", "Bash", "go test ./internal/p -count=1", ""},
		{"a finding", "Bash", "go run ./cmd/finding -title x", ""},
	} {
		if message := call("reviewer", allowed.tool, allowed.command, allowed.path); message != "" {
			t.Errorf("a review session was refused %s: %s", allowed.name, message)
		}
	}
}
