package loop

import (
	"fmt"
	"slices"
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

// findingsWorld is an emptied plan whose register holds open findings no row names.
type findingsWorld struct {
	claimedAfterCycleWorld
	waiting []string
}

func (w *findingsWorld) FindingsToFile() ([]string, error) { return w.waiting, nil }

// TestARunEndsWaitingNeverCompleteWhileWorkIsHeldOrFindingsAwaitFiling holds a run that cannot go on to saying what it
// waits for (SQA-271): an empty plan with open findings to file ended "plan complete" with nothing run. A run finding
// its work held by another worker at the top of a round, when its worker exits, or after the cycle an emptied plan
// started ends waiting with no row parked and exactly its workers run, 0, 1 and 0; and an empty plan with findings to
// file ends waiting for them, naming them, with no measurement and no worker run.
func TestARunEndsWaitingNeverCompleteWhileWorkIsHeldOrFindingsAwaitFiling(t *testing.T) {
	t.Parallel()
	for name, held := range map[string]struct {
		world interface {
			World
			calls() int
		}
		workers int
	}{
		"top of a round":    {&waitingDispatchWorld{}, 0},
		"a worker's exit":   {&claimedAfterWorkerWorld{}, 1},
		"after a cycle ran": {&claimedAfterCycleWorld{improvingWorld: improvingWorld{published: &scorecard.Record{Commit: "b"}}}, 0},
	} {
		outcome, err := Run(held.world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 2})
		if err != nil || outcome.Reason != ReasonWorkWaiting || len(outcome.Parked) != 0 || held.world.calls() != held.workers {
			t.Errorf("%s: a run finding its work held ended %+v (%v) after %d worker run(s), want waiting after %d", name, outcome, err, held.world.calls(), held.workers)
		}
	}
	waiting := []string{"SQA-271 (P2): Keep scheduled work under the existing supervisor's authority and budgets"}
	findings := &findingsWorld{claimedAfterCycleWorld: claimedAfterCycleWorld{improvingWorld: improvingWorld{published: &scorecard.Record{Commit: "b"}}}, waiting: waiting}
	outcome, err := Run(findings, Config{MaxAttemptsPerStep: 1, MaxInvocations: 2})
	if err != nil || outcome.Reason != reasonFindingsWaiting || !slices.Equal(outcome.Findings, waiting) || findings.measured || findings.call != 0 {
		t.Errorf("an empty plan with findings to file ended %+v (%v), measured %v, after %d worker run(s); want waiting for them, named, with nothing run",
			outcome, err, findings.measured, findings.call)
	}
}

func (w *fakeWorld) calls() int { return w.call }
