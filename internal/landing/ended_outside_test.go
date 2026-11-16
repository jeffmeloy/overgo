package landing

import (
	"bufio"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// The settings a process of this test binary reads to land the fixture's change (TestFixtureLanding): the fixture's
// repository and the card's turn it waits for.
const (
	fixtureLandingRepository = "OVERGO_TEST_FIXTURE_LANDING_REPOSITORY"
	fixtureLandingTurn       = "OVERGO_TEST_FIXTURE_LANDING_TURN"
)

// TestFixtureLanding is the process TestALandingEndedFromOutsideLeavesItsAttempt starts to land the fixture's change
// and end from outside: its log is its output. It passes at once in any other run.
func TestFixtureLanding(t *testing.T) {
	t.Parallel()
	repository, turn := os.Getenv(fixtureLandingRepository), os.Getenv(fixtureLandingTurn)
	if repository == "" {
		return
	}
	laneTurn = func() (string, error) { return turn, nil }
	lanes = nil
	_, err := Run(t.Context(), Options{Repo: repository, Ref: "row/do", Log: os.Stdout,
		Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")})
	t.Log(err)
}

// TestALandingEndedFromOutsideLeavesItsAttempt ends a landing from outside while it waits for the card's turn at its
// check, as the cycle-lanes/do landing was ended at 18:01 on 2026-11-15 and left no entry in landing_waits.json, so the
// day's first-attempt rate and gate hours read better than the day was. While the landing runs, its record is written
// and left out of the week's pace, as it has no outcome yet; once its process is gone, the week reads it as a refused
// attempt and the day as a commit attempt failed at verify, the stage it had reached.
func TestALandingEndedFromOutsideLeavesItsAttempt(t *testing.T) {
	// Serial: it names the lane turn the tests take, not the host's own.
	const turnName = "overgo-test-landing-ended-outside"
	turn := laneTurn
	laneTurn = func() (string, error) { return turnName, nil }
	t.Cleanup(func() { laneTurn = turn })
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n\nfunc a() string { return \"base\" }\n"})
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Steps[0].Verify = plan.CardCheckSetting + " go test ./internal/p -run '^TestA$' -count=1"
	})
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "the row's check")
	fixture.write("internal/p/p.go", "package p\n\nfunc a() string { return \"shipped\" }\n")
	fixture.write("internal/p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestA(t *testing.T) {\n\tif a() != \"shipped\" {\n\t\tt.Fatal(a())\n\t}\n}\n")
	held, let := make(chan struct{}), make(chan struct{})
	var group sync.WaitGroup
	group.Go(heldTurn(t, held, let))
	<-held
	t.Cleanup(func() {
		close(let)
		group.Wait()
	})

	landing := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestFixtureLanding$")
	landing.Env = append(os.Environ(), fixtureLandingRepository+"="+fixture.repo, fixtureLandingTurn+"="+turnName)
	landing.Stderr = os.Stderr
	output := testcheck.Must(landing.StdoutPipe())(t)
	testcheck.NoError(t, landing.Start())
	lines, waiting := bufio.NewScanner(output), false
	for !waiting && lines.Scan() {
		waiting = strings.Contains(lines.Text(), "the checkout is free")
	}
	if !waiting {
		t.Fatalf("the landing ended, %v, without waiting for the card's turn", landing.Wait())
	}
	var runs []spent
	testcheck.NoError(t, jsonfile.Decode(filepath.Join(fixture.repo, processlock.StateDirectory, spentRecord), &runs))
	if running := testcheck.Must(WeekPace(fixture.repo, time.Now()))(t); len(runs) != 1 || !runs[0].Ended.IsZero() || running.RefusedShare != 0 {
		t.Errorf("while the landing waits its record reads %+v and the week's pace %+v; want one run with no end, left out of the pace", runs, running)
	}

	testcheck.NoError(t, landing.Process.Kill())
	_ = landing.Wait()
	now := time.Now()
	if pace := testcheck.Must(WeekPace(fixture.repo, now))(t); pace.RefusedShare != 1 {
		t.Errorf("the week's pace after the landing was ended from outside reads %+v; want its one attempt refused", pace)
	}
	if day := testcheck.Must(DayDelivery(t.Context(), fixture.repo, now, ""))(t); day.Refused["verify"] != 1 || day.GateHours <= 0 {
		t.Errorf("the day after the landing was ended from outside reads %d attempts failed at verify over %.4f gate hours; want one, with its time",
			day.Refused["verify"], day.GateHours)
	}
}
