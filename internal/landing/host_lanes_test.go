package landing

import (
	"io"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
)

// TestALandingsLanesRideOnTheCardTurnItTakes holds a landing's lanes on the card to one wait for the card's turn
// (card-handover/lane-turn): the landing takes the turn once, and the lanes' processes, started in the landing's
// environment (laneEnvironment), ride on its place, so none of their tests waits for the card on its own; each of a
// lane's tests had waited out a picture of a render beside it, and a landing's lanes waited 6.2 hours on 2026-11-12.
// Once the lanes end, the processes the landing starts ride on nothing.
func TestALandingsLanesRideOnTheCardTurnItTakes(t *testing.T) {
	// Serial: it names the lane turn the tests take, not the host's own, and the environment the turn is passed in.
	const name = "overgo-test-landing-card-turn"
	turn := laneTurn
	laneTurn = func() (string, error) { return name, nil }
	t.Cleanup(func() { laneTurn = turn })
	rides := func() bool {
		return slices.ContainsFunc(laneEnvironment(lane{}), func(variable string) bool {
			return strings.HasPrefix(variable, processcontrol.TurnEnvironment+"=") && strings.Contains(variable, name)
		})
	}
	l := &landing{Options: Options{Log: io.Discard, Ref: "row/step"}}
	_, shared := splitLanes([]lane{{name: "device"}})
	testcheck.NoError(t, l.inTurnFor(t.Context(), shared, func() error {
		if !rides() {
			t.Errorf("a lane starts in %q, want it to ride on the landing's card turn", os.Getenv(processcontrol.TurnEnvironment))
		}
		return nil
	}))
	// The landing gives the turn back as it ends (Run), and nothing it starts after rides on it.
	testcheck.NoError(t, l.card())
	if rides() {
		t.Error("a process the landing starts after its end rides on the card turn it gave up")
	}
}

// TestALandingHoldsTheCardForItsCardStagesAlone holds a landing's card turn to the stages that use the card
// (lanes-turn/do): a change reaching a lane on the card is landed while another holds the turn, and when the landing
// lets the checkout go to wait for it, its host stages, the test stage the last of them, have passed, where they had
// held the card idle about ten minutes of each landing while a campaign waited; a step's check on the card runs before
// the host's tests, on a turn of its own (verify-first/do). The turn taken carries the processes the landing starts, the lanes' and the scorecard's alike, and a lane
// takes it again for nothing: in the rehearsal of 2026-11-12 the benchmark took the card per context, waited out a
// render's clips and timed its wait as its load.
func TestALandingHoldsTheCardForItsCardStagesAlone(t *testing.T) {
	// Serial: it names the lane turn and the lanes the tests take, not the host's own, and the environment the turn is
	// passed in.
	const name = "overgo-test-landing-whole-card-turn"
	turn, gated := laneTurn, lanes
	laneTurn = func() (string, error) { return name, nil }
	lanes = []lane{{name: "fixture", packages: []string{"./internal/p"}, command: func(string) []string { return []string{"test", "-count=1", "./internal/p"} }}}
	t.Cleanup(func() { laneTurn, lanes = turn, gated })
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n\nfunc a() string { return \"base\" }\n"})
	fixture.write("internal/p/p.go", "package p\n\nfunc a() string { return \"shipped\" }\n")
	held, let := make(chan struct{}), make(chan struct{})
	var group sync.WaitGroup
	group.Go(heldTurn(t, held, let))
	<-held
	freed := make(chan struct{})
	var said strings.Builder
	var before string
	landed := make(chan error, 1)
	group.Go(func() {
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do",
			Log:     freedLog{freed: sync.OnceFunc(func() { before = said.String(); close(freed) }), said: &said},
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")})
		landed <- err
	})
	select {
	case <-freed:
	case err := <-landed:
		close(let)
		group.Wait()
		t.Fatalf("the landing ended, %v, without waiting for the card's turn", err)
	}
	close(let)
	group.Wait()
	if err := <-landed; err != nil {
		t.Fatalf("the landing refused once the turn was free: %v", err)
	}
	if !strings.Contains(before, "land: test ok") || strings.Contains(before, "land: lane fixture runs") {
		t.Errorf("the landing waited for the card's turn before its host stages passed, or ran its lane without it:\n%s", before)
	}
	var order []string
	for _, stage := range (&landing{verifiers: []string{cudaTestSetting + " go test ./internal/p -run '^TestP$'"}}).checks() {
		order = append(order, stage.name)
	}
	if slices.Index(order, "verify") > slices.Index(order, "test") {
		t.Errorf("a check on the card runs after the host's tests: %v", order)
	}
	rides := func() bool {
		return strings.Contains(os.Getenv(processcontrol.TurnEnvironment), name)
	}
	l := &landing{Options: Options{Log: io.Discard, Ref: "row/step"}}
	testcheck.NoError(t, l.holdCard(t.Context()))
	if !rides() {
		t.Fatalf("the landing's processes start in %q after it took the card's turn, want them to ride on it", os.Getenv(processcontrol.TurnEnvironment))
	}
	_, shared := splitLanes([]lane{{name: "device"}})
	testcheck.NoError(t, l.inTurnFor(t.Context(), shared, func() error {
		if !rides() || l.card == nil {
			t.Error("the lanes ran outside the turn the landing took at its first stage that runs tests")
		}
		return nil
	}))
	testcheck.NoError(t, l.card())
	if rides() {
		t.Error("the landing's processes still ride on the card's turn it gave up")
	}
}

// TestAHostLaneRunsOutsideTheLaneTurn holds a landing whose lanes need no device to taking no lane turn, and one that
// reaches a lane of the device to taking it: with a turn no process can take, the host lane runs and the device's is
// refused by the turn. The turn guards the card, which a host lane does not use, and a landing whose one lane was
// webui once waited 7.99 h for a measurement run's turn.
func TestAHostLaneRunsOutsideTheLaneTurn(t *testing.T) {
	// Serial: it names the lane turn the tests take, not the host's own.
	turn := laneTurn
	laneTurn = func() (string, error) { return "", nil }
	t.Cleanup(func() { laneTurn = turn })
	l := &landing{Options: Options{Log: io.Discard}}
	for _, candidate := range []lane{{name: "webui", hostOnly: true}, {name: "device"}} {
		_, shared := splitLanes([]lane{candidate})
		ran := false
		err := l.inTurnFor(t.Context(), shared, func() error {
			ran = true
			return nil
		})
		if candidate.hostOnly && (err != nil || !ran) {
			t.Errorf("lane %s, which needs no device, took the lane turn (ran %t, %v)", candidate.name, ran, err)
		}
		if !candidate.hostOnly && (err == nil || ran) {
			t.Errorf("lane %s ran without the lane turn that guards the device", candidate.name)
		}
	}
}
