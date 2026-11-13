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

// freedLog is a landing's log that says, once, when the landing let the checkout go, and keeps what it said when said
// is set.
type freedLog struct {
	freed func()
	said  *strings.Builder
}

func (f freedLog) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte("the checkout is free")) {
		f.freed()
	}
	if f.said != nil {
		f.said.Write(p)
	}
	return len(p), nil
}

// TestALandingWaitingOnARunLeavesTheCheckoutFree lands a change whose lane is reached while another holds the lane
// turn, as a measurement run does for hours. The landing lets the checkout go once its host stages pass: the lead
// then changes the file the landing ships and edits another, and lets the turn go. The lane ran on the change as it
// stood, whose test the lead's change would fail; the commit is that change, not the checkout as it then stood; the
// lead's two edits are left uncommitted as edited; and the worktree the stages read is kept in the checkout's process
// state, for the next landing.
func TestALandingWaitingOnARunLeavesTheCheckoutFree(t *testing.T) {
	// Serial: it names the lane turn and the lanes the tests take, not the host's own.
	turn, gated := laneTurn, lanes
	laneTurn = func() (string, error) { return "overgo-test-checkout-free-turn", nil }
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
	group.Go(heldTurn(t, held, let))
	<-held
	freed := make(chan struct{})
	log := freedLog{freed: sync.OnceFunc(func() { close(freed) })}
	landed := make(chan error, 1)
	group.Go(func() {
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: log,
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")})
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
	if trees := fixture.git("worktree", "list", "--porcelain"); strings.Count(trees, "worktree ") != 2 ||
		!strings.Contains(trees, processlock.StateDirectory+"/"+candidateCheckout+"0") {
		t.Fatalf("the worktrees are\n%s\nwant the checkout and the one the stages read, kept in its process state", trees)
	}
}

// landBesideACommit lands a change whose lane is reached while another holds the lane turn, and once the landing lets
// the checkout go takes the repository's commit turn it gave up and commits, as another session's report does. It
// returns the fixture, the commit made, what the landing said and how it ended.
func landBesideACommit(t *testing.T) (fixture *sliceFixture, moved, said string, landing error) {
	// Serial: it names the lane turn and the lanes the tests take, not the host's own.
	turn, gated := laneTurn, lanes
	laneTurn = func() (string, error) { return "overgo-test-commit-turn", nil }
	lanes = []lane{{name: "fixture", packages: []string{"./internal/p"}, command: func(string) []string { return []string{"test", "-count=1", "./internal/p"} }}}
	t.Cleanup(func() { laneTurn, lanes = turn, gated })
	fixture = newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n\nfunc a() string { return \"base\" }\n", "docs/notes.txt": "base\n"})
	fixture.git("config", "core.autocrlf", "false")
	fixture.write("internal/p/p.go", "package p\n\nfunc a() string { return \"shipped\" }\n")

	held, let := make(chan struct{}), make(chan struct{})
	var group sync.WaitGroup
	group.Go(heldTurn(t, held, let))
	<-held
	freed := make(chan struct{})
	var log strings.Builder
	landed := make(chan error, 1)
	group.Go(func() {
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: freedLog{freed: sync.OnceFunc(func() { close(freed) }), said: &log},
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")})
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
	moved = fixture.git("rev-parse", "HEAD")
	close(let)
	group.Wait()
	landing = <-landed
	return fixture, moved, log.String(), landing
}

// TestALandingWaitingOnTheDeviceGivesUpItsCommitTurn holds a landing beside a run to waiting only for the device: while
// it waits, the repository's commit turn is free for a commit that needs no lane. That commit moves the trunk the
// landing's checks would pass on, so the landing refuses, publishes nothing and leaves the trunk as the other commit
// left it.
func TestALandingWaitingOnTheDeviceGivesUpItsCommitTurn(t *testing.T) {
	fixture, moved, _, err := landBesideACommit(t)
	if err == nil || !strings.Contains(err.Error(), "while this waited for the device") {
		t.Fatalf("a landing whose trunk moved during the wait = %v, want it refused", err)
	}
	if head := fixture.git("rev-parse", "HEAD"); head != moved {
		t.Fatalf("the refused landing moved the trunk to %s from %s", head, moved)
	}
}

// TestALandingWaitsForALaneTurnAnotherProcessHolds holds the tests beside a run to a run's lane turn: held by another
// process, the landing waits for it and says so. Held in the test's own process, the landing rode it and went on at
// once, and the commit its test made "during the wait" raced the landing's taking its commit turn back (admission-words
// was refused on 2026-11-13 by TestALandingWaitingOnTheDeviceGivesUpItsCommitTurn).
func TestALandingWaitsForALaneTurnAnotherProcessHolds(t *testing.T) {
	_, _, said, _ := landBesideACommit(t)
	if !strings.Contains(said, "waits for the turn") {
		t.Fatalf("the landing said no wait for the lane turn another process held:\n%s", said)
	}
}

// TestALandingTakesItsCommitTurnBackOnceTheCardIsItsOwn holds the commit turn's release to the card's wait alone: a
// commit made while the landing waited refuses it as the card's turn becomes its own, before a stage that runs tests
// starts. On 2026-11-12 the landing had given the commit turn up for its whole run, and a commit another session made
// 30 minutes into process-identity/start's checks refused it after its lanes.
func TestALandingTakesItsCommitTurnBackOnceTheCardIsItsOwn(t *testing.T) {
	_, _, said, err := landBesideACommit(t)
	if err == nil || !strings.Contains(err.Error(), "card turn: land: ") || !strings.Contains(err.Error(), "while this waited for the device") {
		t.Fatalf("a landing whose trunk moved during its card wait = %v, want it refused at the card's turn", err)
	}
	for _, stage := range []string{"lanes", "scorecard"} {
		if strings.Contains(said, ": "+stage+"\n") {
			t.Fatalf("the landing ran its %s stage before refusing:\n%s", stage, said)
		}
	}
}
