package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/guard"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAnEndedSessionsScratchIsFinished holds the session line to the tmp line's rule with one more keeper, the running
// session (session-leftovers/ended-session): a session folder is held while its session's end is unrecorded, and an
// ended one while it holds a worktree, a location the store registers, a folder named for an open row or a name a
// committed line carries. A folder whose session's end is recorded beside its state, or whose session the checkout's
// hold names with a process that has exited, and which holds none of those, is finished, on a line that deletes.
func TestAnEndedSessionsScratchIsFinished(t *testing.T) {
	t.Parallel()
	root := testcheck.Must(filepath.EvalSymlinks(t.TempDir()))(t)
	at := func(slash string) string { return filepath.Join(root, filepath.FromSlash(slash)) }
	write := func(slash, content string) { testutil.WriteTextFile(t, root, slash, content) }

	repo := madeFolder(t, at("repo"))
	gitFixture(t, repo, "init", "-q", "-b", "master")
	gitFixture(t, repo, "config", "core.autocrlf", "false")
	write("repo/.gitignore", "/.overgo-runtime/\n")
	write("repo/docs/notes.md", "a probe was kept in the scratch of session ended-cited\n")
	testcheck.NoError(t, plan.Save(filepath.Join(repo, filepath.FromSlash(plan.Path)), plan.Plan{Campaign: "hygiene", Lane: "master-lead", Items: []plan.Item{{
		ID: "open-row", Title: "a row of the fixture", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "its step", Status: plan.StatusOpen, Verify: "go version"}},
	}}}))
	gitFixture(t, repo, "add", "-A")
	gitFixture(t, repo, "commit", "-q", "-m", "base")

	sessions := madeFolder(t, at("temporary/claude/C--repo"))
	for _, session := range []string{"running", "done", "ended-row/scratchpad/open-row", "ended-cited", "ended-registered", "ended-worktree", "ended-hold"} {
		madeFolder(t, filepath.Join(sessions, filepath.FromSlash(session)))
	}
	write("temporary/claude/C--repo/done/scratchpad/probe.txt", "notes")
	gitFixture(t, repo, "worktree", "add", "-q", "--detach", filepath.Join(sessions, "ended-worktree", "copy"))
	state := filepath.Join(repo, processlock.StateDirectory)
	for _, session := range []string{"done", "ended-row", "ended-cited", "ended-registered", "ended-worktree"} {
		write("repo/"+processlock.StateDirectory+"/"+guard.SessionState(session)+"-ended", "2026-11-11T00:00:00Z\n")
	}
	// The hold names a session and a process that has exited.
	exited := exec.Command("git", "--version")
	testcheck.NoError(t, exited.Run())
	testcheck.NoError(t, os.WriteFile(filepath.Join(state, guard.WorktreeHolder), []byte("ended-hold\n"+strconv.Itoa(exited.Process.Pid)+"\n"), 0o600))

	h := hygiene{
		checkout: repo, data: madeFolder(t, at("data")), sessions: sessions, temporary: at("temporary"), cache: madeFolder(t, at("cache")),
		executable: at("temporary/loop.exe"),
		stores:     func() ([]leftover, error) { return nil, nil },
		registered: func() ([]string, error) {
			return []string{filepath.Join(sessions, "ended-registered", "model.bin")}, nil
		},
		processes: func() ([]string, error) { return nil, nil },
	}
	got := map[string]leftover{}
	for _, entry := range testcheck.Must(listLeftovers(h, true))(t) {
		if entry.place == "session-scratch" {
			got[filepath.Base(entry.path)] = entry
		}
	}
	for session, want := range map[string]string{
		"running": stateHeld, "done": stateFinished, "ended-row": stateHeld, "ended-cited": stateHeld,
		"ended-registered": stateHeld, "ended-worktree": stateHeld, "ended-hold": stateFinished,
	} {
		entry, listed := got[session]
		if !listed || entry.state != want || entry.action != actionDelete {
			t.Errorf("the session folder %s is %s on a line that does %s, by %q (listed %t); want %s on a line that deletes", session, entry.state, entry.action, entry.rule, listed, want)
		}
	}
}
