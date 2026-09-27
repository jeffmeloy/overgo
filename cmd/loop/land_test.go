package main

import (
	"errors"
	"slices"
	"testing"
)

// fakeRowWorld scripts each step's answer and records the order of calls.
type fakeRowWorld struct {
	gate      string
	landedErr error
	calls     []string
}

func (w *fakeRowWorld) Gate(string, string) (string, error) {
	w.calls = append(w.calls, phaseGate)
	return w.gate, nil
}

func (w *fakeRowWorld) Landed(string) (string, error) {
	w.calls = append(w.calls, phaseLanded)
	return "0123456789abcdef0123456789abcdef01234567", w.landedErr
}

// TestRowSequenceStateMachine binds the row sequence: the gate runs once, a
// refusal stops the sequence before git is read, and a landing is only
// believed from git.
func TestRowSequenceStateMachine(t *testing.T) {
	t.Parallel()
	full := []string{phaseGate, phaseLanded}
	for _, scenario := range []struct {
		name    string
		world   fakeRowWorld
		phase   string
		outcome string
		calls   []string
		failed  bool
	}{
		{name: "gate refusal", world: fakeRowWorld{gate: "land: test: failed tests/packages: p"}, phase: phaseGate, outcome: outcomeRefused, calls: full[:1]},
		{name: "gate success git does not confirm", world: fakeRowWorld{landedErr: errors.New("HEAD is not the landing")}, phase: phaseLanded, outcome: outcomeRefused, calls: full, failed: true},
		{name: "landed", phase: phaseLanded, outcome: outcomeReady, calls: full},
	} {
		world := scenario.world
		outcome, err := landRow(&world, "row/do", "message.txt")
		if (err != nil) != scenario.failed || outcome.Phase != scenario.phase || outcome.Outcome != scenario.outcome ||
			!slices.Equal(world.calls, scenario.calls) {
			t.Fatalf("%s: outcome=%+v err=%v calls=%v", scenario.name, outcome, err, world.calls)
		}
		if (outcome.Outcome == outcomeReady) != (outcome.Commit != "") {
			t.Fatalf("%s: commit %q does not match a confirmed landing", scenario.name, outcome.Commit)
		}
	}
}
