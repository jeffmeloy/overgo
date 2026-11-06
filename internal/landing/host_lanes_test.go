package landing

import (
	"io"
	"testing"
)

// TestAHostLaneRunsOutsideTheLaneTurn holds a landing whose lanes need no device to taking no lane turn, and one that
// reaches a lane of the device to taking it: with a turn no process can take, the host lane runs and the device's is
// refused by the turn. The turn guards the card, which a host lane does not use, and a landing whose one lane was
// webui once waited 7.99 h for a measurement run's turn.
func TestAHostLaneRunsOutsideTheLaneTurn(t *testing.T) {
	// Serial: it names the lane turn the tests take, not the host's own.
	turn := laneTurn
	laneTurn = ""
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
