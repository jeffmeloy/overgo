package landing

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// reportWait is how many times the check's run with the change the test waits for the landing's word.
const reportWait = 10

// overrunLog is a landing's log that keeps what it is told, from the stages and from the undo run's clock alike: it
// says once when the check has passed with the change, with the wall from its stage's start, and once when the undo
// run is said to outlast twice its run with the change.
type overrunLog struct {
	mu           sync.Mutex
	said         strings.Builder
	verifying    time.Time
	wall         time.Duration
	passed, told chan struct{}
	pass, tell   func()
}

// newOverrunLog is an overrunLog that has said nothing.
func newOverrunLog() *overrunLog {
	o := &overrunLog{passed: make(chan struct{}), told: make(chan struct{})}
	o.pass = sync.OnceFunc(func() {
		o.wall = time.Since(o.verifying)
		close(o.passed)
	})
	o.tell = sync.OnceFunc(func() { close(o.told) })
	return o
}

func (o *overrunLog) Write(p []byte) (int, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	line := string(p)
	switch {
	case strings.HasPrefix(line, "land: stage ") && strings.HasSuffix(strings.TrimSpace(line), ": verify"):
		o.verifying = time.Now()
	case strings.HasPrefix(line, "verify: pass") && !o.verifying.IsZero():
		o.pass()
	case strings.Contains(line, "it may wait on what only the change sends"):
		o.tell()
	}
	return o.said.Write(p)
}

// passedWall is the wall from the verify stage's start to the check's pass with the change.
func (o *overrunLog) passedWall() time.Duration {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.wall
}

func (o *overrunLog) String() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.said.String()
}

// TestAnUndoRunThatOutlastsTwiceItsPassingWallIsSaid holds the undo run to saying a check's run without the change
// that outlasts twice its run with it (waiter-bound/do): on 2026-11-15 turn-holder/do's undo run of
// TestAWaiterSaysEachNewHolder, the waiter restored without its watch, waited on a line the watch never sent and held
// the gate 32 minutes before its probes were found and ended by hand; inside the machine no clock ends work, so the
// clock says it. The fixture's code before the change waits for a file only the test makes, and its change returns
// at once: the check passes with the change, the landing says, at twice that wall, that the run without it may wait on
// what only the change sends and in which tree; the test then makes the file, the run fails on its assertion, and the
// change lands with that failure as its undo evidence.
func TestAnUndoRunThatOutlastsTwiceItsPassingWallIsSaid(t *testing.T) {
	t.Parallel()
	goOn := filepath.Join(t.TempDir(), "go-on")
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n\nimport (\n\t\"errors\"\n\t\"os\"\n\t\"time\"\n)\n\n" +
		"func wait() error {\n\tfor {\n\t\tif _, err := os.Stat(" + strconv.Quote(goOn) + "); err == nil {\n" +
		"\t\t\treturn errors.New(\"waited for the file the test makes\")\n\t\t}\n\t\ttime.Sleep(time.Millisecond)\n\t}\n}\n"})
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Steps[0].Verify = "go test ./internal/p -run '^TestA$' -count=1"
	})
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "the row's check")
	fixture.write("internal/p/p.go", "package p\n\nfunc wait() error { return nil }\n")
	fixture.write("internal/p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestA(t *testing.T) {\n\tif err := wait(); err != nil {\n\t\tt.Fatal(err)\n\t}\n}\n")
	log := newOverrunLog()
	landed := make(chan error, 1)
	go func() {
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: log,
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")})
		landed <- err
	}()
	// The test waits for the landing's word as long as ten times the check's run with the change: a gate that says
	// nothing fails here, its run ended with the test's context, where it would wait for ever.
	passed, due := log.passed, (<-chan time.Time)(nil)
	for told := false; !told; {
		select {
		case <-passed:
			passed, due = nil, time.After(reportWait*log.passedWall())
		case <-due:
			t.Fatalf("the landing did not say its undo run outlasted twice its run with the change within ten times that run, %s:\n%s",
				reportWait*log.passedWall(), log.String())
		case <-log.told:
			told = true
		case err := <-landed:
			t.Fatalf("the landing ended %v before saying its undo run outlasted twice its run with the change:\n%s", err, log.String())
		}
	}
	if said := log.String(); !strings.Contains(said, "twice the") || !strings.Contains(said, "undone-0") {
		t.Errorf("the landing said its undo run outlasted its bound without the bound or the tree it runs in:\n%s", said)
	}
	testcheck.NoError(t, os.WriteFile(goOn, nil, 0o600))
	if err := <-landed; err != nil {
		t.Fatalf("once the wait ended the landing ended %v; want it landed:\n%s", err, log.String())
	}
	if body := fixture.git("log", "-1", "--format=%B"); !strings.Contains(body, "fails: ") || strings.Contains(body, "Undo-Evidence: none") {
		t.Errorf("the landed commit does not hold the undo run's failure as its evidence:\n%s", body)
	}
}
