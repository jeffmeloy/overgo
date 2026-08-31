package runrecord

import (
	"testing"
)

// TestCanonicalExecutionOutcomeContract holds the vocabulary closed:
// exactly ten members, valid by membership alone. Run documents keep
// their executed-run subset: a refusal never produces a run, it
// produces a refused attempt.
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
}
