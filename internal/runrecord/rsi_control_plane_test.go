package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestDeterministicRSIControlPlane proves the runrecord slice of the control
// plane: causal derivation keeps one root across triggers, terminal attempt
// evidence admits only complete records, a measured interaction reduction
// wins only when no counter silently worsens, resource comparison refuses an
// empty or incomparable denominator, and rollout plans admit only the
// registered deterministic cohort hash.
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

	// Interaction efficiency: a reduction wins only with every counter
	// covered and no counter silently worsened.
	baselineTrace := EfficiencyTrace{
		Surface: SurfaceAgent, Task: "plane-task",
		Work:     InteractionWork{ModelTurns: 8, Commits: 3, Retries: 1},
		Result:   id(artifact.KindEvidence, "trace result"),
		Evidence: id(artifact.KindEvidence, "trace evidence"),
	}
	improved := baselineTrace
	improved.Work = InteractionWork{ModelTurns: 5, Commits: 3, Retries: 1}
	covered := EfficiencyCounterNames()
	comparison, err := CompareEfficiencyTraces(improved, baselineTrace, covered, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !comparison.Win || len(comparison.Worsened) != 0 {
		t.Fatalf("measured reduction did not win cleanly: %+v", comparison)
	}
	shifted := baselineTrace
	shifted.Work = InteractionWork{ModelTurns: 5, Commits: 3, Retries: 4}
	shiftedComparison, err := CompareEfficiencyTraces(shifted, baselineTrace, covered, nil)
	if err != nil {
		t.Fatal(err)
	}
	if shiftedComparison.Win || len(shiftedComparison.Worsened) == 0 {
		t.Fatalf("shifted work claimed a win: %+v", shiftedComparison)
	}
	if _, err := CompareEfficiencyTraces(improved, baselineTrace, covered[:1], nil); err == nil ||
		!strings.Contains(err.Error(), "not covered") {
		t.Fatalf("uncovered counter compared: %v", err)
	}

	// Resource comparison refuses an empty denominator.
	if _, err := NewResourceFitnessComparison(nil); err == nil {
		t.Fatal("an empty resource comparison was admitted")
	}

	// Rollout plans admit only the registered deterministic hash and assign
	// the same unit identically forever.
	authorities := []artifact.ID{
		id(artifact.KindEvidence, "counterfactual"), id(artifact.KindEvidence, "cohort"),
		id(artifact.KindEvidence, "observation"), id(artifact.KindEvidence, "rollback"),
	}
	build := func(hashVersion uint16) (RolloutPlan, error) {
		return NewRolloutPlan(RolloutPlan{
			Baseline: id(artifact.KindRecipe, "baseline"), Candidate: id(artifact.KindRecipe, "candidate"),
			Admission: authorities[0], CohortKey: "request-digest",
			HashVersion: hashVersion, CohortShare: 5000, CohortEvidence: authorities[1],
			Observation: RolloutObservation{
				Metric: "exact-match", MinObservations: 2, Evidence: authorities[2],
			},
			Rollback: authorities[3],
		})
	}
	if _, err := build(RolloutHashVersion + 1); err == nil ||
		!strings.Contains(err.Error(), "hash version") {
		t.Fatalf("unregistered hash version admitted: %v", err)
	}
	plan, err := build(RolloutHashVersion)
	if err != nil {
		t.Fatal(err)
	}
	first, err := plan.Assign("workload-000")
	if err != nil {
		t.Fatal(err)
	}
	for range 3 {
		if again, err := plan.Assign("workload-000"); err != nil || again != first {
			t.Fatalf("cohort assignment drifted: (%s, %v)", again, err)
		}
	}
}
