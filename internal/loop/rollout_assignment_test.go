package loop

import (
	"fmt"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func rolloutAssignmentPlan(t *testing.T, share uint32) runrecord.RolloutPlan {
	t.Helper()
	plan, err := runrecord.NewRolloutPlan(runrecord.RolloutPlan{
		Baseline:       testutil.ArtifactID(t, artifact.KindRecipe, "assignment-baseline"),
		Candidate:      testutil.ArtifactID(t, artifact.KindRecipe, "assignment-candidate"),
		Admission:      testutil.ArtifactID(t, artifact.KindEvidence, "assignment-counterfactual"),
		CohortKey:      "request-digest",
		HashVersion:    runrecord.RolloutHashVersion,
		CohortShare:    share,
		CohortEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "assignment-cohort-derivation"),
		Observation: runrecord.RolloutObservation{
			Metric:          "exact-match",
			MinObservations: 64,
			Evidence:        testutil.ArtifactID(t, artifact.KindEvidence, "assignment-observation-derivation"),
		},
		Rollback: testutil.ArtifactID(t, artifact.KindEvidence, "assignment-rollback-decision"),
	})
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

// TestRolloutAssignmentIsStable pins the registered version-1 assignment:
// the arm serving one workload unit is a pure function of the plan's
// content identity, its cohort key, and the unit, so the same unit keeps
// the same arm across retries, replay from re-parsed plan bytes, peer
// placement, restart, and call order; the cohort share bounds hold, every
// arm named is one of the plan's two recipes, and an unidentified plan or
// unbounded unit refuses.
func TestRolloutAssignmentIsStable(t *testing.T) {
	plan := rolloutAssignmentPlan(t, 5000)
	units := make([]string, 0, 200)
	for index := range 200 {
		units = append(units, fmt.Sprintf("workload-%03d", index))
	}
	arms := make(map[string]artifact.ID, len(units))
	candidates := 0
	for _, unit := range units {
		arm, err := plan.Assign(unit)
		if err != nil {
			t.Fatal(err)
		}
		if arm != plan.Baseline && arm != plan.Candidate {
			t.Fatalf("unit %s assigned foreign arm %s", unit, arm)
		}
		if arm == plan.Candidate {
			candidates++
		}
		arms[unit] = arm
	}
	if candidates == 0 || candidates == len(units) {
		t.Fatalf("half-share plan assigned %d/%d candidates", candidates, len(units))
	}

	// Retry: repeated assignment in reversed order changes nothing.
	for index := len(units) - 1; index >= 0; index-- {
		if arm, err := plan.Assign(units[index]); err != nil || arm != arms[units[index]] {
			t.Fatalf("retry reassigned %s: (%s, %v)", units[index], arm, err)
		}
	}
	// Restart and peer placement: an independently reconstructed plan is the
	// same content identity and assigns identically.
	rebuilt := rolloutAssignmentPlan(t, 5000)
	if rebuilt.ID != plan.ID {
		t.Fatalf("reconstructed plan drifted: %s != %s", rebuilt.ID, plan.ID)
	}
	// Replay: a plan re-parsed from its committed bytes assigns identically.
	batch, err := plan.Batch("rollout/assignment-fixture")
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := runrecord.ParseRolloutPlan(batch.Contents[0].Data)
	if err != nil {
		t.Fatal(err)
	}
	for _, unit := range units {
		if arm, err := rebuilt.Assign(unit); err != nil || arm != arms[unit] {
			t.Fatalf("restart reassigned %s: (%s, %v)", unit, arm, err)
		}
		if arm, err := replayed.Assign(unit); err != nil || arm != arms[unit] {
			t.Fatalf("replay reassigned %s: (%s, %v)", unit, arm, err)
		}
	}

	full := rolloutAssignmentPlan(t, runrecord.RolloutCohortDenominator)
	sliver := rolloutAssignmentPlan(t, 1)
	fullCandidates, sliverCandidates := 0, 0
	for _, unit := range units {
		if arm, err := full.Assign(unit); err != nil || arm == full.Candidate {
			if err != nil {
				t.Fatal(err)
			}
			fullCandidates++
		}
		if arm, err := sliver.Assign(unit); err != nil || arm == sliver.Candidate {
			if err != nil {
				t.Fatal(err)
			}
			sliverCandidates++
		}
	}
	if fullCandidates != len(units) {
		t.Fatalf("full-share plan assigned %d/%d candidates", fullCandidates, len(units))
	}
	if sliverCandidates >= candidates {
		t.Fatalf("one-basis-point share assigned %d candidates; half share assigned %d", sliverCandidates, candidates)
	}

	if _, err := plan.Assign(""); err == nil || !strings.Contains(err.Error(), "bounded workload unit") {
		t.Fatalf("empty unit assigned: %v", err)
	}
	var unidentified runrecord.RolloutPlan
	if _, err := unidentified.Assign(units[0]); err == nil ||
		!strings.Contains(err.Error(), "identified rollout plan") {
		t.Fatalf("unidentified plan assigned: %v", err)
	}
}
