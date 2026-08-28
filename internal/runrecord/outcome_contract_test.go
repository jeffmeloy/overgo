package runrecord

import (
	"testing"
)

// TestCanonicalExecutionOutcomeContract holds the vocabulary closed:
// exactly ten members, valid by membership alone, with every gate
// step and lane outcome mapping totally onto it -- reuse maps to
// success as evidence provenance and an unavailable lane fails by
// doctrine. Run documents keep their executed-run subset: a refusal
// never produces a run, it produces a refused attempt.
func TestCanonicalExecutionOutcomeContract(t *testing.T) {
	if len(CanonicalOutcomes) != 10 {
		t.Fatalf("canonical vocabulary holds %d members, want 10", len(CanonicalOutcomes))
	}
	seen := map[Outcome]bool{}
	for _, outcome := range CanonicalOutcomes {
		if outcome == "" || seen[outcome] || !ValidOutcome(outcome) {
			t.Fatalf("member %q empty, repeated, or invalid", outcome)
		}
		seen[outcome] = true
	}
	if ValidOutcome("almost-succeeded") {
		t.Fatal("foreign outcome admitted")
	}
	for _, step := range []StepOutcome{StepSucceeded, StepFailed, StepCancelled, StepSkipped, StepInapplicable, StepReused} {
		if _, ok := StepOutcomeCanonical(step); !ok {
			t.Fatalf("gate step outcome %q has no canonical mapping", step)
		}
	}
	if canonical, _ := StepOutcomeCanonical(StepReused); canonical != OutcomeSucceeded {
		t.Fatal("reuse is evidence provenance and must map to success")
	}
	for _, lane := range []LaneOutcome{LanePassed, LaneFailed, LaneUnavailable, LaneEmpty} {
		if _, ok := LaneOutcomeCanonical(lane); !ok {
			t.Fatalf("lane outcome %q has no canonical mapping", lane)
		}
	}
	if canonical, _ := LaneOutcomeCanonical(LaneUnavailable); canonical != OutcomeFailed {
		t.Fatal("unavailable must fail; it never passes silently")
	}
}
