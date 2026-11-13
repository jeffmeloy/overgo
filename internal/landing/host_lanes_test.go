package landing

import (
	"io"
	"os"
	"slices"
	"strings"
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
	if rides() {
		t.Error("a process the landing starts after its lanes rides on the card turn it gave up")
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
