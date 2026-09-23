package main

import (
	"errors"
	"slices"
	"testing"
)

// fakeRowWorld scripts each step's answer and records the order of calls.
type fakeRowWorld struct {
	claim, preflight, gate, lanes string
	report                        *laneReport
	landedErr                     error
	pending                       []string
	calls                         []string
}

func (w *fakeRowWorld) Claim(string) (string, error) {
	w.calls = append(w.calls, phaseClaim)
	return w.claim, nil
}

func (w *fakeRowWorld) Preflight(string) (string, error) {
	w.calls = append(w.calls, phasePreflight)
	return w.preflight, nil
}

func (w *fakeRowWorld) Gate(string, string) (string, error) {
	w.calls = append(w.calls, phaseGate)
	return w.gate, nil
}

func (w *fakeRowWorld) Landed(string) (string, error) {
	w.calls = append(w.calls, "landed")
	return "0123456789abcdef0123456789abcdef01234567", w.landedErr
}

func (w *fakeRowWorld) AwaitLanes() (*laneReport, string, error) {
	w.calls = append(w.calls, phaseLanes)
	return w.report, w.lanes, nil
}

func (w *fakeRowWorld) PendingReview() ([]string, error) {
	w.calls = append(w.calls, phaseReview)
	return w.pending, nil
}

// TestRowSequenceStateMachine binds the row sequence: every phase runs once
// and in order, the first refusal stops it before anything later runs -- no
// gate without a claim, no gate over preflight findings -- a landing is only
// believed from git, a failed lane is its own outcome, and the outcome names
// whether the landing still waits on review answers.
func TestRowSequenceStateMachine(t *testing.T) {
	full := []string{phaseClaim, phasePreflight, phaseGate, "landed", phaseLanes, phaseReview}
	for _, scenario := range []struct {
		name    string
		world   fakeRowWorld
		phase   string
		outcome string
		calls   []string
		failed  bool
	}{
		{name: "unclaimable row", world: fakeRowWorld{claim: "claimed by another worker"}, phase: phaseClaim, outcome: outcomeRefused, calls: full[:1]},
		{name: "preflight findings", world: fakeRowWorld{preflight: "magics FAIL"}, phase: phasePreflight, outcome: outcomeRefused, calls: full[:2]},
		{name: "gate blocker", world: fakeRowWorld{gate: "blocker: test-owners"}, phase: phaseGate, outcome: outcomeRefused, calls: full[:3]},
		{name: "gate success git does not confirm", world: fakeRowWorld{landedErr: errors.New("HEAD is not the completion")}, phase: phaseGate, outcome: outcomeRefused, calls: full[:4], failed: true},
		{name: "deferred lane failed", world: fakeRowWorld{lanes: "Deferred validation failed"}, phase: phaseLanes, outcome: outcomeLanesFailed, calls: full[:5]},
		{name: "landed with review pending", world: fakeRowWorld{pending: []string{"suite-cost suite-cost:pkg: 9s"}}, phase: phaseReview, outcome: outcomeReviewPending, calls: full},
		{name: "landed and clear", phase: phaseReview, outcome: outcomeReady, calls: full},
	} {
		world := scenario.world
		outcome, err := landRow(&world, "row/do", "message.txt")
		if (err != nil) != scenario.failed || outcome.Phase != scenario.phase || outcome.Outcome != scenario.outcome ||
			!slices.Equal(world.calls, scenario.calls) {
			t.Fatalf("%s: outcome=%+v err=%v calls=%v", scenario.name, outcome, err, world.calls)
		}
		if landed := slices.Contains(world.calls, phaseLanes); landed != (outcome.Commit != "") {
			t.Fatalf("%s: commit %q does not match a confirmed landing", scenario.name, outcome.Commit)
		}
	}
}
