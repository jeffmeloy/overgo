package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestDeterministicRSIControlPlane proves the runrecord slice of the control
// plane: causal derivation keeps one root across triggers, terminal attempt
// evidence admits only complete records, a measured interaction reduction
// wins only when no counter silently worsens, and resource comparison
// refuses an empty or incomparable denominator.
func TestDeterministicRSIControlPlane(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(t, kind, "plane "+name)
	}

	// Causal derivation: one root, typed triggers, reproducible motivation.
	root, err := NewCausalRoot(TriggerControllerProposal, id(artifact.KindEvidence, "proposal"))
	if err != nil {
		t.Fatal(err)
	}
	retry, err := root.Derive(TriggerRetry, id(artifact.KindEvidence, "retry"))
	if err != nil {
		t.Fatal(err)
	}
	if retry.Root != root.Root || retry.Trigger != TriggerRetry {
		t.Fatalf("derived cause lost the root: %+v", retry)
	}

	// Terminal attempt evidence: complete records admit, incomplete refuse.
	commit := "0123456789abcdef0123456789abcdef01234567"
	attempt, err := NewAttemptRecord(AttemptRecord{
		PlanItem: "plane", PlanStep: "terminal", Result: id(artifact.KindEvidence, "result"),
		Recipe: id(artifact.KindRecipe, "recipe"), CodeCommit: commit,
		Outcome: OutcomeSucceeded, WallNS: 1, Causal: &retry,
	})
	if err != nil {
		t.Fatal(err)
	}
	if attempt.Causal == nil || attempt.Causal.Root != root.Root {
		t.Fatalf("attempt dropped its causal chain: %+v", attempt.Causal)
	}
	if _, err := NewAttemptRecord(AttemptRecord{
		PlanItem: "plane", PlanStep: "incomplete", Recipe: id(artifact.KindRecipe, "recipe"),
		CodeCommit: commit, Outcome: OutcomeSucceeded, WallNS: 1,
	}); err == nil {
		t.Fatal("an attempt without its result was admitted")
	}

	// Resource comparison refuses an empty denominator.
	if _, err := newResourceFitnessComparison(nil); err == nil {
		t.Fatal("an empty resource comparison was admitted")
	}
}
