package loop

import (
	"strings"
	"testing"
)

// fakeWorld scripts the driver's environment: a queue of plan steps, a worker
// whose behavior per invocation is programmed, and recorded park calls.
type fakeWorld struct {
	steps    []Step
	behavior []func(w *fakeWorld, step Step, feedback string) string // returns verify failure after this invocation, "" = advance
	call     int
	failure  string
	parked   []string
	paused   bool
	prompts  []string
	feedback []string
}

func (w *fakeWorld) Current() (Step, bool, error) {
	if len(w.steps) == 0 {
		return Step{}, false, nil
	}
	return w.steps[0], true, nil
}

func (w *fakeWorld) Prompt(step Step) (string, error) {
	w.prompts = append(w.prompts, step.Key())
	return "TASK " + step.Key(), nil
}

func (w *fakeWorld) RunWorker(step Step, prompt, feedback string) (string, error) {
	w.feedback = append(w.feedback, feedback)
	if w.call < len(w.behavior) {
		w.failure = w.behavior[w.call](w, step, feedback)
	}
	w.call++
	return "worker transcript tail", nil
}

func (w *fakeWorld) Verify(Step) (string, error) { return w.failure, nil }

func (w *fakeWorld) Park(step Step, reason string) error {
	w.parked = append(w.parked, step.Key()+": "+reason)
	return nil
}

func (w *fakeWorld) Paused() bool { return w.paused }

// TestLoopDriverCycle pins the whole contract: the driver advances through
// worker-committed steps to plan completion, feeds verify failures back into
// retries, parks a stuck step after the attempt budget instead of spinning,
// stops on the invocation budget, and honors the pause switch -- all with no
// concept of a turn: worker exits are just events.
func TestLoopDriverCycle(t *testing.T) {
	// Two steps: the first needs two attempts (first verify fails, feedback
	// flows into attempt two), the second lands first try.
	world := &fakeWorld{
		steps: []Step{{Item: "alpha", ID: "do"}, {Item: "beta", ID: "do"}},
		behavior: []func(w *fakeWorld, step Step, feedback string) string{
			func(w *fakeWorld, _ Step, feedback string) string {
				if feedback != "" {
					t.Fatalf("first attempt carried stale feedback: %q", feedback)
				}
				return "TestAlpha FAILED: got 4 want 5"
			},
			func(w *fakeWorld, _ Step, feedback string) string {
				if !strings.Contains(feedback, "got 4 want 5") {
					t.Fatalf("retry did not carry the verify failure: %q", feedback)
				}
				w.steps = w.steps[1:] // worker committed through the gate
				return ""
			},
			func(w *fakeWorld, _ Step, _ string) string {
				w.steps = w.steps[1:]
				return ""
			},
		},
	}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 3, MaxInvocations: 10})
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Reason != ReasonPlanComplete || outcome.Invocations != 3 || len(outcome.Parked) != 0 {
		t.Fatalf("outcome = %+v, want plan-complete after 3 invocations", outcome)
	}

	// A step that never advances parks after the attempt budget with the last
	// failure in the reason -- loudly terminal, never a silent skip.
	stuck := &fakeWorld{
		steps: []Step{{Item: "stuck", ID: "do"}},
		behavior: []func(w *fakeWorld, step Step, feedback string) string{
			func(*fakeWorld, Step, string) string { return "oracle NOT satisfied" },
			func(*fakeWorld, Step, string) string { return "oracle NOT satisfied" },
		},
	}
	outcome, err = Run(stuck, Config{MaxAttemptsPerStep: 2, MaxInvocations: 10})
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Reason != ReasonParked || len(stuck.parked) != 1 ||
		!strings.Contains(stuck.parked[0], "oracle NOT satisfied") {
		t.Fatalf("stuck step not parked with evidence: %+v %v", outcome, stuck.parked)
	}

	// Invocation budget is a hard mechanical stop.
	budget := &fakeWorld{
		steps: []Step{{Item: "granite", ID: "do"}},
		behavior: []func(w *fakeWorld, step Step, feedback string) string{
			func(*fakeWorld, Step, string) string { return "still failing" },
		},
	}
	outcome, err = Run(budget, Config{MaxAttemptsPerStep: 100, MaxInvocations: 1})
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Reason != ReasonBudget || outcome.Invocations != 1 {
		t.Fatalf("budget stop = %+v", outcome)
	}

	// Pause marker wins before any work.
	paused := &fakeWorld{steps: []Step{{Item: "x", ID: "do"}}, paused: true}
	outcome, err = Run(paused, Config{MaxAttemptsPerStep: 1, MaxInvocations: 1})
	if err != nil || outcome.Reason != ReasonPaused || outcome.Invocations != 0 {
		t.Fatalf("pause = (%+v, %v)", outcome, err)
	}

	// Green-verifier-but-uncommitted feeds a commit instruction back.
	uncommitted := &fakeWorld{
		steps: []Step{{Item: "done-not-shipped", ID: "do"}},
		behavior: []func(w *fakeWorld, step Step, feedback string) string{
			func(*fakeWorld, Step, string) string { return "" },
			func(w *fakeWorld, _ Step, feedback string) string {
				if !strings.Contains(feedback, "commit through the gate") {
					t.Fatalf("uncommitted-green feedback missing: %q", feedback)
				}
				w.steps = nil
				return ""
			},
		},
	}
	if _, err := Run(uncommitted, Config{MaxAttemptsPerStep: 3, MaxInvocations: 5}); err != nil {
		t.Fatal(err)
	}

	// Zero budgets refuse to run.
	if _, err := Run(&fakeWorld{}, Config{}); err == nil {
		t.Fatal("unbounded driver accepted")
	}
}
