package landing

import (
	"errors"
	"io"
	"sync"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/quietmachine"
	"overgo/internal/testcheck"
)

// TestALandingBesideARunWaitsOnlyForTheDevice holds a landing beside a
// measurement run to waiting for the device alone. The run's measurement and
// fill hold the host's lane turn (HoldLaneTurn, cmd/loop), and the landing's
// host stages run before its lanes take any turn; its lanes wait for the turn
// holding no side of the quiet machine, so a timed cell the run takes alone
// meanwhile is admitted. Once the run lets the turn go the lanes run, the
// landing's side of the machine taken again. Every build, test and commit was
// once held for a run by a written rule.
func TestALandingBesideARunWaitsOnlyForTheDevice(t *testing.T) {
	// Serial: it names the lane turn the tests take, not the host's own.
	turn := laneTurn
	laneTurn = "overgo-test-lane-turn"
	t.Cleanup(func() { laneTurn = turn })
	const machine = "overgo-test-quiet-machine-lanes"
	ctx := t.Context()
	l := &landing{Options: Options{Log: io.Discard}}
	release := testcheck.Must(l.holdQuiet(ctx, machine))(t)
	l.quiet, l.quietMachine = release, machine

	held, let := make(chan struct{}), make(chan struct{})
	ran := make(chan bool, 1)
	var group sync.WaitGroup
	group.Go(heldTurn(t, held, let))
	<-held
	group.Go(func() {
		if err := l.inLaneTurn(ctx, func() error {
			hold, busy := processcontrol.HoldResource(machine)
			if busy == nil {
				_ = hold()
			}
			ran <- errors.Is(busy, processcontrol.ErrResourceBusy)
			return nil
		}); err != nil {
			t.Error(err)
		}
	})
	measuring, _, err := quietmachine.Await(ctx, machine, true, noWait)
	testcheck.NoError(t, err)
	select {
	case <-ran:
		t.Fatal("the landing's lanes ran while the run held the lane turn")
	default:
	}
	testcheck.NoError(t, measuring())
	close(let)
	if heldAgain := <-ran; !heldAgain {
		t.Fatal("the landing's lanes ran without its side of the quiet machine")
	}
	group.Wait()
	if l.quiet == nil {
		t.Fatal("the landing kept no side of the quiet machine after its lanes")
	}
	testcheck.NoError(t, l.quiet())
}
