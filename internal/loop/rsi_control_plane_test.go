package loop

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestDeterministicRSIControlPlane runs one prototype through the loop's
// slice of the control plane as one flowing scenario: an admitted worker
// materializes into an exact strategy identity, its attempts carry the
// causal root through a resource comparison, the promoted candidate rides a
// deterministic rollout cohorting, a live regression contains and rolls the
// activation back atomically, and the published finding admits a
// re-mechanized reentry — every transition externally visible through the
// store, none through a second control path.
func TestDeterministicRSIControlPlane(t *testing.T) {
	// Admitted prototype: compiled strategy identity, terminal attempt
	// evidence, resource comparison, reproducible motivation.
	fixture := newStrategyExperimentTestFixture(t)
	winner, comparison, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline, fixture.candidates, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(winner.Candidates) == 0 || len(comparison.Evidence) < len(fixture.candidates) {
		t.Fatalf("comparison closed nothing: candidates=%d evidence=%d", len(winner.Candidates), len(comparison.Evidence))
	}
	attempt, err := runrecord.RequireAttemptRecord(fixture.ctx, fixture.store, fixture.candidates[0].Attempt)
	if err != nil {
		t.Fatal(err)
	}
	if attempt.Causal == nil || attempt.Causal.Root != fixture.causalRoot.Root {
		t.Fatalf("attempt lost the causal root: %+v", attempt.Causal)
	}

	// Model and recipe materialization on one shared store for the rest of
	// the plane.
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "promotion-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "rsi-control-plane/facts",
		Artifacts: []artifact.Descriptor{
			{ID: modelID, Size: 4096},
			{ID: testutil.ArtifactID(t, artifact.KindProfile, "promotion-profile")},
			{ID: testutil.ArtifactID(t, artifact.KindModelDefinition, "promotion-definition")},
		},
	}); err != nil {
		t.Fatal(err)
	}
	baseline := promotionDefinition(t, modelID, recipe.ResidencyHostReference)
	candidate := promotionDefinition(t, modelID, recipe.ResidencyHostCache)
	for name, definition := range map[string]recipe.Definition{
		"baseline": baseline, "candidate": candidate,
	} {
		if _, _, err := modelrecipe.PublishCandidate(
			ctx, store, "rsi-control-plane/"+name, definition,
		); err != nil {
			t.Fatal(err)
		}
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, baseline,
		promotionVerification(t, store, baseline.ID, "rsi-control-plane/baseline-verification"),
		recipe.EvidenceVerified, "baseline in service",
	); err != nil {
		t.Fatal(err)
	}

	// Deterministic rollout cohorting: the same unit lands in the same arm
	// on every assignment, and an independently reconstructed plan is the
	// same content identity assigning identically.
	planAuthority := func(name string) artifact.ID {
		id := testutil.ArtifactID(t, artifact.KindEvidence, name)
		testutil.PublishArtifact(t, store, id)
		return id
	}
	buildPlan := func() runrecord.RolloutPlan {
		plan, planErr := runrecord.NewRolloutPlan(runrecord.RolloutPlan{
			Baseline: baseline.ID, Candidate: candidate.ID,
			Admission: planAuthority("plane-counterfactual"), CohortKey: "request-digest",
			HashVersion: runrecord.RolloutHashVersion, CohortShare: 5000,
			CohortEvidence: planAuthority("plane-cohort-derivation"),
			Observation: runrecord.RolloutObservation{
				Metric: "exact-match", MinObservations: 2,
				Evidence: planAuthority("plane-observation-derivation"),
			},
			Rollback: planAuthority("plane-rollback-decision"),
		})
		if planErr != nil {
			t.Fatal(planErr)
		}
		return plan
	}
	plan := buildPlan()
	rebuilt := buildPlan()
	if rebuilt.ID != plan.ID {
		t.Fatalf("reconstructed rollout plan drifted: %s != %s", rebuilt.ID, plan.ID)
	}
	assigned := map[string]artifact.ID{}
	candidates := 0
	for index := 0; index < 64; index++ {
		unit := fmt.Sprintf("workload-%03d", index)
		arm, assignErr := plan.Assign(unit)
		if assignErr != nil {
			t.Fatal(assignErr)
		}
		assigned[unit] = arm
		if arm == plan.Candidate {
			candidates++
		}
		if again, againErr := rebuilt.Assign(unit); againErr != nil || again != arm {
			t.Fatalf("cohorting is not deterministic for %s: (%s, %v)", unit, again, againErr)
		}
	}
	if candidates == 0 || candidates == len(assigned) {
		t.Fatalf("half-share plan assigned %d/%d candidates", candidates, len(assigned))
	}

	// Promotion, live regression, containment, and atomic rollback: the
	// restored activation is the baseline, the retired candidate is
	// superseded, and the finding cites its causes.
	if err := modelrecipe.ActivateCapability(
		ctx, store, candidate,
		promotionVerification(t, store, candidate.ID, "rsi-control-plane/candidate-verification"),
		recipe.EvidenceVerified, "candidate promoted",
	); err != nil {
		t.Fatal(err)
	}
	finding, err := RollbackQuarantinedCandidate(
		ctx, store, modelID, recipe.TaskInference, candidate.ID,
		publishBreakerTransition(t, store, 2),
		promotionFailedVerification(t, store, candidate.ID, "rsi-control-plane/failed-verification"),
		promotionVerification(t, store, baseline.ID, "rsi-control-plane/reactivation-verification"),
		"sustained live regression",
	)
	if err != nil {
		t.Fatal(err)
	}
	restored, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !active || restored.Definition.ID != baseline.ID {
		t.Fatalf("rollback restored (%s, %v, %v)", restored.Definition.ID, active, err)
	}
	status, published, err := modelrecipe.Status(ctx, store, candidate.ID)
	if err != nil || !published || status != recipe.StatusSuperseded {
		t.Fatalf("retired candidate = (%s, %v, %v)", status, published, err)
	}
	content, found, err := artifact.ReadContent(ctx, store, finding)
	if err != nil || !found {
		t.Fatalf("rollback finding unpublished: (%v, %v)", found, err)
	}
	var causal LiveRollbackRecord
	if err := json.Unmarshal(content.Data, &causal); err != nil {
		t.Fatal(err)
	}
	if causal.Retired != candidate.ID || causal.Restored != baseline.ID {
		t.Fatalf("finding lost its causes: %+v", causal)
	}

	// Reproducible reentry: fresh mechanism, evaluation, and plan evidence
	// introduced after the quarantine admit the candidate back.
	mechanism := publishReentryEvidence(t, store, "plane-changed-mechanism")
	reevaluation := publishReentryEvidence(t, store, "plane-new-evaluation")
	freshPlan := buildPlanAfterQuarantine(t, store, baseline.ID, candidate.ID)
	admitted, err := AdmitQuarantineReentry(ctx, store, QuarantineReentry{
		Candidate: candidate.ID, Finding: finding, Mechanism: mechanism,
		Evaluation: reevaluation, Plan: freshPlan,
		Decider: "operator:jeffm", Reason: "mechanism replaced and re-evaluated",
	})
	if err != nil {
		t.Fatal(err)
	}
	if has, hasErr := store.HasContent(ctx, admitted); hasErr != nil || !has {
		t.Fatalf("reentry admission unpublished: (%v, %v)", has, hasErr)
	}
}

// buildPlanAfterQuarantine commits a fresh rollout plan whose evidence is
// introduced after the quarantine finding, the reentry admission contract.
func buildPlanAfterQuarantine(
	t *testing.T, store *overgodb.Store, baseline, candidate artifact.ID,
) artifact.ID {
	t.Helper()
	ctx := context.Background()
	authority := func(name string) artifact.ID {
		id := testutil.ArtifactID(t, artifact.KindEvidence, name)
		testutil.PublishArtifact(t, store, id)
		return id
	}
	plan, err := runrecord.NewRolloutPlan(runrecord.RolloutPlan{
		Baseline: baseline, Candidate: candidate,
		Admission: authority("plane-fresh-counterfactual"), CohortKey: "request-digest",
		HashVersion: runrecord.RolloutHashVersion, CohortShare: 100,
		CohortEvidence: authority("plane-fresh-cohort-derivation"),
		Observation: runrecord.RolloutObservation{
			Metric: "exact-match", MinObservations: 64,
			Evidence: authority("plane-fresh-observation-derivation"),
		},
		Rollback: authority("plane-fresh-rollback-decision"),
	})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := plan.Batch("rsi-control-plane/fresh-plan")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	return plan.ID
}
