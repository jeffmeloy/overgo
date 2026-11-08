package loop

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// commandWorld is a fakeWorld that works a run step itself: lands says whether its command writes the record, which
// lands the step, or leaves the check failing.
type commandWorld struct {
	fakeWorld
	lands bool
	ran   []string
}

func (w *commandWorld) RunCommand(step Step) (string, error) {
	w.ran = append(w.ran, step.Run)
	if w.failure = "the record " + step.Record + " was not written"; w.lands {
		w.steps, w.failure = w.steps[1:], ""
	}
	return "command output", nil
}

// TestARunStepRunsWithoutAWorker holds a run step to its command: in the week to Nov 7 a model sat through
// vision-bridge's 54-minute capture and its GPU-hour bridge run, adding nothing to them. The driver runs a run step's
// command with no worker invocation and goes on to the code step behind it, which a worker takes; a command that leaves
// its check failing is run again with the failure fed back and parked at the attempt budget, still with no worker.
func TestARunStepRunsWithoutAWorker(t *testing.T) {
	t.Parallel()
	capture := Step{Item: "bridge", ID: "capture", Run: "go run ./cmd/validate -capture", Record: "docs/verification/capture.json"}
	landed := &commandWorld{lands: true, fakeWorld: fakeWorld{steps: []Step{capture, {Item: "code", ID: "do"}},
		behavior: []func(*fakeWorld, Step, string) string{func(w *fakeWorld, _ Step, _ string) string { w.steps = w.steps[1:]; return "" }}}}
	outcome := testcheck.Must(Run(landed, Config{MaxAttemptsPerStep: 2, MaxInvocations: 1}))(t)
	if outcome.Reason != ReasonPlanComplete || !slices.Equal(landed.ran, []string{capture.Run}) || landed.call != 1 || outcome.Invocations != 1 {
		t.Fatalf("a run step then a code step ended %+v, running %q and calling the worker %d times; want the command run once and the worker once, for the code step",
			outcome, landed.ran, landed.call)
	}

	stuck := &commandWorld{fakeWorld: fakeWorld{steps: []Step{capture}}}
	outcome = testcheck.Must(Run(stuck, Config{MaxAttemptsPerStep: 2, MaxInvocations: 1}))(t)
	if outcome.Reason != ReasonParked || len(stuck.ran) != 2 || stuck.call != 0 || outcome.Invocations != 0 || len(stuck.parked) != 1 {
		t.Fatalf("a run step whose record is never written ended %+v, running its command %d times and the worker %d; want it parked after two runs and no worker",
			outcome, len(stuck.ran), stuck.call)
	}
}
