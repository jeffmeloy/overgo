package landing

import (
	"bytes"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/processlock"
)

// freedLog is a landing's log that says, once, when the landing let the checkout go.
type freedLog struct{ freed func() }

func (f freedLog) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte("the checkout is free")) {
		f.freed()
	}
	return len(p), nil
}

// TestALandingWaitingOnARunLeavesTheCheckoutFree lands a change whose lane is reached while another holds the lane
// turn, as a measurement run does for hours. The landing lets the checkout go once its host stages pass: the lead
// then changes the file the landing ships and edits another, and lets the turn go. The lane ran on the change as it
// stood, whose test the lead's change would fail; the commit is that change, not the checkout as it then stood; the
// lead's two edits are left uncommitted as edited; and the worktree the lanes read is gone.
func TestALandingWaitingOnARunLeavesTheCheckoutFree(t *testing.T) {
	// Serial: it names the lane turn and the lanes the tests take, not the host's own.
	turn, gated := laneTurn, lanes
	laneTurn = "overgo-test-checkout-free-turn"
	lanes = []lane{{name: "fixture", packages: []string{"./internal/p"}, command: func(string) []string { return []string{"test", "-count=1", "./internal/p"} }}}
	t.Cleanup(func() { laneTurn, lanes = turn, gated })
	answer := func(value string) string {
		return "package p\n\nfunc a() string { return \"" + value + "\" }\n"
	}
	const check = "package p\n\nimport \"testing\"\n\nfunc TestA(t *testing.T) {\n\tif a() != \"shipped\" {\n\t\tt.Fatal(a())\n\t}\n}\n"
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": answer("base"), "docs/notes.txt": "base\n"})
	fixture.git("config", "core.autocrlf", "false")
	fixture.write("internal/p/p.go", answer("shipped"))
	fixture.write("internal/p/p_test.go", check)

	held, let := make(chan struct{}), make(chan struct{})
	var group sync.WaitGroup
	group.Go(func() {
		if err := HoldLaneTurn(func() error {
			close(held)
			<-let
			return nil
		}); err != nil {
			t.Error(err)
		}
	})
	<-held
	freed := make(chan struct{})
	log := freedLog{freed: sync.OnceFunc(func() { close(freed) })}
	landed := make(chan error, 1)
	group.Go(func() {
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: log,
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: none.\n")})
		landed <- err
	})
	select {
	case <-freed:
	case err := <-landed:
		close(let)
		group.Wait()
		t.Fatalf("the landing ended, %v, without letting the checkout go while the turn was held", err)
	}
	fixture.write("internal/p/p.go", answer("edited"))
	fixture.write("docs/notes.txt", "the lead's next row\n")
	close(let)
	group.Wait()
	if err := <-landed; err != nil {
		t.Fatalf("the landing refused once the checkout was edited: %v", err)
	}
	if committed := fixture.git("show", "HEAD:internal/p/p.go"); !strings.Contains(committed, "shipped") {
		t.Fatalf("the commit holds %q, want the change as it stood when the checkout was let go", committed)
	}
	if held := fixture.read("internal/p/p.go"); !strings.Contains(held, "edited") || fixture.read("docs/notes.txt") != "the lead's next row\n" {
		t.Fatal("the lead's edits made while the landing waited were not left as edited")
	}
	if status := fixture.git("status", "--porcelain"); strings.Count(status, "\n") != 1 || !strings.Contains(status, "docs/notes.txt") || !strings.Contains(status, "internal/p/p.go") {
		t.Fatalf("the checkout's uncommitted work = %q, want the lead's two edits alone", status)
	}
	if trees := fixture.git("worktree", "list", "--porcelain"); strings.Count(trees, "worktree ") != 1 {
		t.Fatalf("the worktree the lanes read was left:\n%s", trees)
	}
}

// TestALandingWaitingOnTheDeviceGivesUpItsCommitTurn holds a landing beside a run to waiting only for the device: while
// its lanes wait, the repository's commit turn is free for a commit that needs no lane. That commit moves the trunk the
// landing's checks passed on, so when the lanes finish the landing refuses, publishes nothing and leaves the trunk as
// the other commit left it.
func TestALandingWaitingOnTheDeviceGivesUpItsCommitTurn(t *testing.T) {
	// Serial: it names the lane turn and the lanes the tests take, not the host's own.
	turn, gated := laneTurn, lanes
	laneTurn = "overgo-test-commit-turn"
	lanes = []lane{{name: "fixture", packages: []string{"./internal/p"}, command: func(string) []string { return []string{"test", "-count=1", "./internal/p"} }}}
	t.Cleanup(func() { laneTurn, lanes = turn, gated })
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n\nfunc a() string { return \"base\" }\n", "docs/notes.txt": "base\n"})
	fixture.git("config", "core.autocrlf", "false")
	fixture.write("internal/p/p.go", "package p\n\nfunc a() string { return \"shipped\" }\n")

	held, let := make(chan struct{}), make(chan struct{})
	var group sync.WaitGroup
	group.Go(func() {
		if err := HoldLaneTurn(func() error {
			close(held)
			<-let
			return nil
		}); err != nil {
			t.Error(err)
		}
	})
	<-held
	freed := make(chan struct{})
	log := freedLog{freed: sync.OnceFunc(func() { close(freed) })}
	landed := make(chan error, 1)
	group.Go(func() {
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: log,
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: none.\n")})
		landed <- err
	})
	select {
	case <-freed:
	case err := <-landed:
		close(let)
		group.Wait()
		t.Fatalf("the landing ended, %v, without waiting on the lane turn", err)
	}
	path := filepath.Join(fixture.repo, processlock.StateDirectory, turnLock)
	lock, err := processlock.Acquire(path, clioptions.PrivateFileMode)
	if err == nil {
		err = lock.Close()
	}
	if err != nil {
		close(let)
		group.Wait()
		t.Fatalf("the commit turn stayed held while the landing waited on the device: %v", err)
	}
	fixture.git("commit", "--allow-empty", "-m", "another commit")
	moved := fixture.git("rev-parse", "HEAD")
	close(let)
	group.Wait()
	if err := <-landed; err == nil || !strings.Contains(err.Error(), "while this waited for the device") {
		t.Fatalf("a landing whose trunk moved during the wait = %v, want it refused", err)
	}
	if head := fixture.git("rev-parse", "HEAD"); head != moved {
		t.Fatalf("the refused landing moved the trunk to %s from %s", head, moved)
	}
}
