package guard

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"overgo/internal/processlock"
)

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
