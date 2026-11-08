package loop

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// ladderWorld is a fakeWorld whose workers stand in a list: passes says, by step and then by worker, whether a run
// lands the step; used records each run as the step's item and the worker it went to; a step set aside leaves the
// queue, as a parked row leaves dispatch.
type ladderWorld struct {
	fakeWorld
	passes map[string][]bool
	level  int
	used   []string
}

func (w *ladderWorld) Workers() int        { return len(w.passes[w.steps[0].Item]) }
func (w *ladderWorld) UseWorker(level int) { w.level = level }

func (w *ladderWorld) RunWorker(step Step, _, _ string) (string, error) {
	w.used, w.failure = append(w.used, fmt.Sprintf("%s:%d", step.Item, w.level)), "the check still fails"
	if w.passes[step.Item][w.level] {
		w.steps, w.failure = w.steps[1:], ""
	}
	return "tail", nil
}

func (w *ladderWorld) Park(step Step, reason string) error {
	w.parked, w.steps = append(w.parked, step.Key()+": "+reason), w.steps[1:]
	return nil
}

// TestAStepMovesUpTheWorkersBeforeItIsSetAside holds a hard step to the stronger workers before it waits on anyone:
// overgo is to run on a much weaker model, and this week's missing attention gates and the 1e21 gradient from a zero
// start needed a strong one. A step the first worker cannot pass in its attempts goes to the second, which lands it; a
// step neither passes is set aside after the last worker's attempts, with a finding, and the next step runs, starting
// again with the first worker.
func TestAStepMovesUpTheWorkersBeforeItIsSetAside(t *testing.T) {
	t.Parallel()
	world := &ladderWorld{
		passes:    map[string][]bool{"gates": {false, true}, "gradient": {false, false}, "next": {true, true}},
		fakeWorld: fakeWorld{steps: []Step{{Item: "gates", ID: "do"}, {Item: "gradient", ID: "do"}, {Item: "next", ID: "do"}}},
	}
	outcome := testcheck.Must(Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 10}))(t)
	want := []string{"gates:0", "gates:0", "gates:1", "gradient:0", "gradient:0", "gradient:1", "gradient:1", "next:0"}
	if !slices.Equal(world.used, want) || len(world.parked) != 1 || outcome.Reason != ReasonPlanComplete {
		t.Fatalf("the runs went %v, ending %+v with %d set aside; want %v, the gradient step set aside and the plan worked to its end", world.used, outcome, len(world.parked), want)
	}
}
