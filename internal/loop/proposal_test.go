package loop

import (
	"testing"
)

// proposalWorld scripts a drained plan fed by a proposal queue: each
// admitted proposal becomes one step whose worker either advances it
// (pass) or never does (park). It implements World and ProposalSource.
type proposalWorld struct {
	queue    []string        // item ids to admit, in order
	passes   map[string]bool // item id -> the worker advances it
	current  string          // dispatched item id, "" = drained
	admitted []string
	blocked  []string
	parked   []string
}

func (w *proposalWorld) Current() (Step, bool, error) {
	if w.current == "" {
		return Step{}, false, nil
	}
	return Step{Item: w.current, ID: "do"}, true, nil
}
func (w *proposalWorld) Prompt(Step) (string, error) { return "do it", nil }
func (w *proposalWorld) RunWorker(step Step, prompt, feedback string) (string, error) {
	if w.passes[step.Item] {
		w.current = "" // the worker advanced the row through the gate
	}
	return "tail", nil
}
func (w *proposalWorld) Verify(Step) (string, error) { return "verify failed", nil }
func (w *proposalWorld) Park(step Step, reason string) error {
	w.parked = append(w.parked, step.Key())
	return nil
}
func (w *proposalWorld) Paused() bool { return false }

func (w *proposalWorld) AdmitNext() (string, bool, error) {
	if len(w.queue) == 0 {
		return "", false, nil
	}
	item := w.queue[0]
	w.queue = w.queue[1:]
	w.admitted = append(w.admitted, item)
	w.current = item
	return item, true, nil
}
func (w *proposalWorld) Block(step Step, reason string) error {
	w.blocked = append(w.blocked, step.Key())
	w.current = "" // blocked out of dispatch; the plan drains again
	return nil
}

// TestProposalClosureConsumesQueueUntilEmpty pins the closure: a
// drained plan admits proposals one at a time, an advancing proposal
// row resets saturation, and an empty queue ends the run as
// plan-complete.
func TestProposalClosureConsumesQueueUntilEmpty(t *testing.T) {
	world := &proposalWorld{
		queue:  []string{"p1", "p2"},
		passes: map[string]bool{"p1": true, "p2": true},
	}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 10, SaturationLimit: 2})
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Reason != ReasonPlanComplete || len(world.admitted) != 2 || len(world.parked) != 0 {
		t.Fatalf("outcome = %+v, admitted %v, parked %v", outcome, world.admitted, world.parked)
	}
}

// TestProposalClosureSaturates pins the stop condition: consecutive
// proposal rows parking without measured improvement block out of
// dispatch and, at the configured limit, end the run as saturated
// instead of consuming the rest of the queue.
func TestProposalClosureSaturates(t *testing.T) {
	world := &proposalWorld{
		queue:  []string{"p1", "p2", "p3"},
		passes: map[string]bool{},
	}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 10, SaturationLimit: 2})
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Reason != ReasonSaturated {
		t.Fatalf("outcome = %+v", outcome)
	}
	if len(world.admitted) != 2 || len(world.blocked) != 2 || len(world.queue) != 1 {
		t.Fatalf("admitted %v blocked %v remaining %v", world.admitted, world.blocked, world.queue)
	}
}

// TestProposalClosureResetsOnMeasuredImprovement pins the streak: a
// parked proposal row counts toward saturation, an advancing one
// resets the count, so mixed outcomes keep consuming the queue.
func TestProposalClosureResetsOnMeasuredImprovement(t *testing.T) {
	world := &proposalWorld{
		queue:  []string{"p1", "p2", "p3", "p4"},
		passes: map[string]bool{"p2": true, "p4": true},
	}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 20, SaturationLimit: 2})
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Reason != ReasonPlanComplete || len(world.admitted) != 4 {
		t.Fatalf("outcome = %+v, admitted %v", outcome, world.admitted)
	}
	if len(world.blocked) != 2 {
		t.Fatalf("blocked = %v", world.blocked)
	}
}

// TestProposalClosureDisabledKeepsOldSemantics pins the default: with
// SaturationLimit zero -- or a world without a proposal source -- a
// drained plan ends the run exactly as before.
func TestProposalClosureDisabledKeepsOldSemantics(t *testing.T) {
	world := &proposalWorld{queue: []string{"p1"}, passes: map[string]bool{"p1": true}}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 10})
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Reason != ReasonPlanComplete || len(world.admitted) != 0 {
		t.Fatalf("outcome = %+v, admitted %v", outcome, world.admitted)
	}
}
