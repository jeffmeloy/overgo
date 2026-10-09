package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/plan"
)

// sessionCheckout makes the working directory a repository whose plan holds
// one open row, owned by the lane master-lead, with no gate committing, and
// returns a writer of its files.
func sessionCheckout(t *testing.T) func(name, content string) {
	t.Helper()
	root := t.TempDir()
	t.Chdir(root)
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	t.Setenv(plan.AutomationMaintenanceEnvironment, "")
	t.Setenv(plan.AutomationRoleEnvironment, "master-lead")
	// The gate runs this test, and a gate that is committing lets every turn end.
	committing := gateCommitting
	gateCommitting = func() bool { return false }
	t.Cleanup(func() { gateCommitting = committing })
	write := func(name, content string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	git := func(arguments ...string) {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", append([]string{"-c", "core.hooksPath=", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid"}, arguments...)...)
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("fixture git %v: %v %s", arguments, err, output)
		}
	}
	git("init", "-q")
	if err := os.MkdirAll("docs", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(plan.Path, plan.Plan{Campaign: "fixture", Doctrine: "fixture", Lane: "master-lead", Items: []plan.Item{{
		ID: "row", Title: "row", Owner: "master-lead", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "row", Status: plan.StatusOpen, Verify: "true"}},
	}}}); err != nil {
		t.Fatal(err)
	}
	git("add", "-A")
	git("commit", "-qm", "plan")
	return write
}

// stopHook is the stop gate's answer to the session: 0 lets its turn end.
func stopHook(session string) int { return runStop(fmt.Sprintf(`{"session_id": %q}`, session)) }

// charged names the files the stop gate charges a session's turn with.
func charged(t *testing.T, session string) []string {
	t.Helper()
	base, known := readTurnBase(session)
	current, err := currentDirtyFacts()
	if !known || err != nil {
		t.Fatalf("session %s has no snapshot to be judged against (%v)", session, err)
	}
	var paths []string
	for _, fact := range turnCreatedDirt(base.Facts, current) {
		paths = append(paths, fact.Path)
	}
	slices.Sort(paths)
	return paths
}

// TestOverlappingSessionsKeepTheirOwnTurnSnapshots begins two turns in two
// sessions of one checkout and holds each to the snapshot it took. The first
// session begins, writes a file, and the second begins: with one snapshot for
// the checkout the second prompt overwrote the first session's, and the first
// was no longer charged with the file it wrote. Each session's snapshot is its
// own: the first is still charged with its file, the second is not charged
// with a file that was there when it began, and a commit boundary in one
// session moves that session's snapshot and leaves the other's. The second
// session takes the worktree over from the first, as a session does that
// continues another's work in its checkout, so both are implementing sessions
// there, one after the other.
func TestOverlappingSessionsKeepTheirOwnTurnSnapshots(t *testing.T) {
	write := sessionCheckout(t)
	first, second := hookInput{Session: "session-one", Prompt: "work the row"}, hookInput{Session: "session-two", Prompt: "Take over this worktree and work another row"}
	beginTurn(first)
	write("first.txt", "the first session's work\n")
	beginTurn(second)
	if code := stopHook(second.Session); code == 0 {
		t.Fatal("a session of the lane's role under an open row may end its turn")
	}
	if got := charged(t, second.Session); len(got) != 0 {
		t.Fatalf("the second session is charged with %v, which was there when its turn began", got)
	}
	write("second.txt", "the second session's work\n")
	if got := charged(t, first.Session); !slices.Contains(got, "first.txt") {
		t.Fatalf("the first session is charged with %v after the second began a turn; want the file it wrote among them", got)
	}
	if got := charged(t, second.Session); !slices.Equal(got, []string{"second.txt"}) {
		t.Fatalf("the second session is charged with %v; want only the file written after its turn began", got)
	}
	runPostCommit(fmt.Sprintf(`{"session_id": %q, "tool_input": {"command": "go run ./cmd/gate -plan row/do"}}`, first.Session))
	if got := charged(t, first.Session); len(got) != 0 {
		t.Fatalf("after its commit boundary the first session is charged with %v; want nothing", got)
	}
	if got := charged(t, second.Session); !slices.Equal(got, []string{"second.txt"}) {
		t.Fatalf("the first session's commit boundary moved the second session's snapshot: it is charged with %v", got)
	}
	// A hook the harness gives no session keeps the one snapshot sessions shared.
	beginTurn(hookInput{Prompt: "a harness that names no session"})
	if _, known := readTurnBase(""); !known || len(charged(t, second.Session)) != 1 {
		t.Fatal("a hook with no session did not keep the shared snapshot, or took a session's")
	}
}

// TestReviewSessionIsNotHeldAndAnyOtherRoleIs holds the stop gate's one
// exemption to the session that declared itself a review session, in its first
// message. Every other session is held, whatever its role: the lane's role and
// no role, which is the lane's, are held to the open row and to the work their
// turn left; another implementing role, which the plan gives no row, may end a
// turn that left nothing and is held to the work a turn did leave, since it
// can write code. The review session, in the same checkout with the same files
// uncommitted, is held to nothing, and stays a review session across a commit
// boundary and a later prompt; and a session that did not begin as one cannot
// become one.
func TestReviewSessionIsNotHeldAndAnyOtherRoleIs(t *testing.T) {
	write := sessionCheckout(t)
	if !beginTurn(hookInput{Session: "reviewer", Prompt: "Review session: read the last three commits and say what they miss"}) {
		t.Fatal("a session whose first message opens with the declaration is not a review session")
	}
	if beginTurn(hookInput{Session: "implementer", Prompt: "work the row"}) || beginTurn(hookInput{Session: "implementer", Prompt: "review session, from now on"}) {
		t.Fatal("a session that began as an implementing session became a review session")
	}
	if beginTurn(hookInput{Prompt: "review session"}) {
		t.Fatal("a hook the harness gives no session declared the shared snapshot a review session's")
	}
	for _, held := range []struct {
		name, role string
		idle       bool
	}{
		{"the lane's role", "master-lead", false},
		{"no role", "", false},
		{"another implementing role", "another-lead", true},
	} {
		t.Setenv(plan.AutomationRoleEnvironment, held.role)
		session := "session-" + held.role
		beginTurn(hookInput{Session: session, Prompt: "take over this worktree and work"})
		if ended := stopHook(session) == 0; ended != held.idle {
			t.Errorf("%s, its turn having left nothing: the turn may end %t, want %t", held.name, ended, held.idle)
		}
		write(session+".txt", "work this session's turn left\n")
		if stopHook(session) == 0 {
			t.Errorf("%s may end a turn that left work uncommitted", held.name)
		}
	}
	// The files the other sessions left are in the checkout the review session shares.
	for _, hook := range []string{`{"session_id": "reviewer"}`, `{"session_id": "reviewer", "prompt": "and the commit before those"}`} {
		runPostCommit(`{"session_id": "reviewer", "tool_input": {"command": "git log --grep cmd/gate"}}`)
		runPrompt(hook)
		if code := runStop(hook); code != 0 {
			t.Fatalf("the review session was held (%d) with other sessions' files uncommitted", code)
		}
	}
	if base, known := readTurnBase("reviewer"); !known || !base.Review {
		t.Fatalf("the review session's state after a commit boundary and a prompt: %+v, %t", base, known)
	}
}
