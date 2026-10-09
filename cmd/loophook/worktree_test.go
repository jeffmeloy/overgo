package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"overgo/internal/guard"
	"overgo/internal/plan"
)

// TestSecondImplementingSessionIsSentToAWorktreeOfItsOwn begins implementing
// sessions in one worktree and holds the worktree to the first. A landing
// takes the checkout's whole tree, so the second session's files would join
// the first one's commit: the second is told who holds the worktree, lands
// nothing there, edits no file the repository tracks, and is held to no row
// and to no file, the first one's among them; the first is unchanged by it. A
// review session takes no worktree and sends no one away. In a worktree of
// its own the second session is admitted: it holds that one, edits there and
// is held to its row. The first worktree stays the first session's through a
// turn a valve lets end, and is given back when a turn of its ends with the
// plan complete and nothing uncommitted: the session that begins next holds
// it, and the first, coming back, is the one sent away. A session whose
// message opens with the takeover declaration takes a worktree from its
// holder.
func TestSecondImplementingSessionIsSentToAWorktreeOfItsOwn(t *testing.T) {
	write := sessionCheckout(t)
	first, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	git := func(arguments ...string) {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", append([]string{"-c", "core.hooksPath=", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid"}, arguments...)...)
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("fixture git %v: %v %s", arguments, err, output)
		}
	}
	// refused says why the guard refuses a session's call in the worktree the
	// test stands in, or nothing when it allows it.
	refused := func(session, tool, command, path string) string {
		t.Helper()
		root, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		call := guard.ToolCall{ToolName: tool, Session: session}
		call.ToolInput.Command, call.ToolInput.Path = command, path
		if path != "" && !filepath.IsAbs(path) {
			call.ToolInput.Path = filepath.Join(root, path)
		}
		message, _ := guard.Verdict(call, root)
		return message
	}
	holder := func(session string) string {
		t.Helper()
		base, known := readTurnBase(session)
		if !known {
			t.Fatalf("session %s has no state", session)
		}
		return base.Holder
	}
	const landing = "go run ./cmd/loop -land row/do -message-file - < message.txt"
	source := filepath.Join("internal", "p", "p.go")

	if !beginTurn(hookInput{Session: "reviewer", Prompt: "Review session: read the last commit"}) {
		t.Fatal("the review session is not one")
	}
	beginTurn(hookInput{Session: "one", Prompt: "work the row"})
	if holder("one") != "" || stopHook("one") == 0 || refused("one", "Edit", "", source) != "" || refused("one", "Bash", landing, "") != "" {
		t.Fatalf("the first implementing session, after a review session, is sent away by %q, or is not held to its row, or is refused its work", holder("one"))
	}

	beginTurn(hookInput{Session: "two", Prompt: "work the row as well"})
	if holder("two") != "one" {
		t.Fatalf("a second implementing session in the worktree is told %q holds it; want the first session", holder("two"))
	}
	write("one.txt", "the first session's work\n")
	if refused("two", "Edit", "", source) == "" || refused("two", "Write", "", filepath.Join("docs", "notes.json")) == "" || refused("two", "Bash", landing, "") == "" {
		t.Error("the second session may edit a file the repository tracks, or land, in the first one's worktree")
	}
	if message := refused("two", "Write", "", filepath.Join(t.TempDir(), "notes.md")); message != "" {
		t.Errorf("the second session is refused a file outside the checkout: %s", message)
	}
	if stopHook("two") != 0 || stopHook("one") == 0 {
		t.Error("the second session is held in the first one's worktree, or the first is no longer held to its row and its file")
	}
	if beginTurn(hookInput{Session: "two", Prompt: "and now?"}); holder("two") != "one" || refused("one", "Edit", "", source) != "" {
		t.Error("asked again, the second session is no longer sent away, or the first is refused its work")
	}

	second := filepath.Join(t.TempDir(), "second")
	git("worktree", "add", "-q", "-b", "second", second, "HEAD")
	t.Chdir(second)
	beginTurn(hookInput{Session: "two", Prompt: "work the row here"})
	if holder("two") != "" || refused("two", "Edit", "", source) != "" || refused("two", "Bash", landing, "") != "" || stopHook("two") == 0 {
		t.Fatalf("in a worktree of its own the second session is sent away by %q, or refused its work, or not held to its row", holder("two"))
	}

	t.Chdir(first)
	// A turn a valve lets end keeps the worktree.
	gateCommitting = func() bool { return true }
	if stopHook("one") != 0 {
		t.Fatal("a committing gate did not let the turn end")
	}
	gateCommitting = func() bool { return false }
	if beginTurn(hookInput{Session: "three", Prompt: "work the row"}); holder("three") != "one" {
		t.Fatalf("after a turn of the first session that a valve let end, the next session is told %q holds the worktree; want the first", holder("three"))
	}
	// A turn that ends with the plan complete and nothing uncommitted gives it back.
	if err := plan.Save(plan.Path, plan.Plan{Campaign: "fixture", Doctrine: "fixture", Lane: "master-lead"}); err != nil {
		t.Fatal(err)
	}
	if stopHook("one") != 0 {
		t.Fatal("the plan complete did not let the turn end")
	}
	if beginTurn(hookInput{Session: "three", Prompt: "work"}); holder("three") != "one" {
		t.Fatal("the first session gave the worktree back with its work uncommitted")
	}
	git("add", "-A")
	git("commit", "-qm", "the row")
	if stopHook("one") != 0 {
		t.Fatal("the plan complete and the work committed did not let the turn end")
	}
	beginTurn(hookInput{Session: "three", Prompt: "work"})
	beginTurn(hookInput{Session: "one", Prompt: "back again"})
	if holder("three") != "" || holder("one") != "three" {
		t.Fatalf("after the first session's turn ended with nothing left to do, the next session is told %q holds the worktree and the first, coming back, %q; want the next to hold it", holder("three"), holder("one"))
	}

	beginTurn(hookInput{Session: "four", Prompt: "Take over this worktree: the session that held it is gone"})
	beginTurn(hookInput{Session: "three", Prompt: "still here"})
	if holder("four") != "" || holder("three") != "four" {
		t.Fatalf("a session that declared a takeover is told %q holds the worktree, and the one it took it from %q", holder("four"), holder("three"))
	}
	if beginTurn(hookInput{Prompt: "a harness that names no session"}); holder("") != "" {
		t.Error("a session the harness does not name was sent away, which no later message of its could undo")
	}
}
