package loop

import (
	"fmt"
	"testing"

	"overgo/internal/scorecard"
)

// claimed is the dispatch's answer for a row another worker holds.
var claimed = fmt.Errorf("%w: claimed by another worker until release", ErrWorkWaiting)

type waitingDispatchWorld struct{ fakeWorld }

func (w *waitingDispatchWorld) Current() (Step, bool, error) {
	return Step{}, false, claimed
}

// claimedAfterWorkerWorld hands its one row to a worker, whose exit finds the row another worker's.
type claimedAfterWorkerWorld struct{ fakeWorld }

func (w *claimedAfterWorkerWorld) Current() (Step, bool, error) {
	if w.call > 0 {
		return Step{}, false, claimed
	}
	return Step{Item: "held", ID: "do"}, true, nil
}

// claimedAfterCycleWorld is an emptied plan whose cycle files a row another worker claims while its lanes run.
type claimedAfterCycleWorld struct {
	improvingWorld
	measured bool
}

func (w *claimedAfterCycleWorld) Measure() (*scorecard.Record, *scorecard.Record, error) {
	w.measured = true
	return w.improvingWorld.Measure()
}

func (w *claimedAfterCycleWorld) Current() (Step, bool, error) {
	if w.measured {
		return Step{}, false, claimed
	}
	return w.fakeWorld.Current()
}

// TestARunThatFindsWorkWaitingSaysSo holds a run that finds its work held by another worker to ending as waiting,
// wherever it finds it: at the top of a round, when a worker exits, or after the cycle an emptied plan started; never as
// a completed plan or an error, and with no row parked and no worker run for a held row.
func TestARunThatFindsWorkWaitingSaysSo(t *testing.T) {
	t.Parallel()
	for name, world := range map[string]interface {
		World
		calls() int
	}{
		"top of a round":    &waitingDispatchWorld{},
		"a worker's exit":   &claimedAfterWorkerWorld{},
		"after a cycle ran": &claimedAfterCycleWorld{improvingWorld: improvingWorld{published: &scorecard.Record{Commit: "b"}}},
	} {
		outcome, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 2})
		if err != nil || outcome.Reason != ReasonWorkWaiting || len(outcome.Parked) != 0 || world.calls() > 1 {
			t.Errorf("%s: a run finding its work held ended %+v (%v) after %d worker run(s), want waiting", name, outcome, err, world.calls())
		}
	}
}

func (w *fakeWorld) calls() int { return w.call }
