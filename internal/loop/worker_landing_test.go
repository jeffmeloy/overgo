package loop

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// landingWorld is a fakeWorld that lands a worker's step whose check passes (LandingWorld); refuse makes its gate
// refuse the landing.
type landingWorld struct {
	fakeWorld
	refuse bool
	landed []string
}

func (w *landingWorld) Land(step Step) (string, error) {
	w.landed = append(w.landed, step.Key())
	if w.refuse {
		return "gate: land: rules: new inline number", errors.New("loop: the gate refused " + step.Key())
	}
	w.steps = w.steps[1:]
	return "landed " + step.Key(), nil
}

// TestTheDriverLandsAWorkersPassedStep holds the driver to landing what a worker passed: a worker of overgo's own works
// with six tools, none of which lands, and was asked to land all the same. A worker that leaves its step's check passing
// has the step landed by the driver, once and with no second worker call; a landing the gate refuses goes back to the
// worker with the gate's words, and a step whose landing is refused at every attempt is parked.
func TestTheDriverLandsAWorkersPassedStep(t *testing.T) {
	t.Parallel()
	passes := func(*fakeWorld, Step, string) string { return "" }
	step := Step{Item: "code", ID: "do"}
	world := &landingWorld{fakeWorld: fakeWorld{steps: []Step{step}, behavior: []func(*fakeWorld, Step, string) string{passes}}}
	outcome := testcheck.Must(Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 3}))(t)
	if outcome.Reason != ReasonPlanComplete || world.call != 1 || !slices.Equal(world.landed, []string{step.Key()}) {
		t.Fatalf("a passed step ended %+v after %d worker calls and landings %v; want it landed once by the driver after one call", outcome, world.call, world.landed)
	}
	refused := &landingWorld{refuse: true, fakeWorld: fakeWorld{steps: []Step{step}, behavior: []func(*fakeWorld, Step, string) string{passes, passes}}}
	outcome = testcheck.Must(Run(refused, Config{MaxAttemptsPerStep: 2, MaxInvocations: 3}))(t)
	if outcome.Reason != ReasonParked || len(refused.landed) != 2 || len(refused.feedback) != 2 || !strings.Contains(refused.feedback[1], "new inline number") {
		t.Fatalf("a refused landing ended %+v after landings %v with feedback %q; want the gate's words fed back and the step parked", outcome, refused.landed, refused.feedback)
	}
}
