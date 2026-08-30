package loop

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const promotionTestCommit = "0123456789abcdef0123456789abcdef01234567"

func promotionVerification(
	t *testing.T, store artifact.Repository, definitionID artifact.ID, key string,
) modelrecipe.Verification {
	t.Helper()
	ctx := context.Background()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: key, OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	environmentBatch, err := environment.Batch(key + "/environment")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, environmentBatch); err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewGateRecord(
		definitionID, environment.ID, promotionTestCommit,
		runrecord.OutcomeSucceeded, "", 1,
		[]runrecord.GateStep{{
			Name: "verify", Phase: runrecord.PhaseValidate,
			Outcome: runrecord.StepSucceeded, DurationNS: 1,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch(key)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	return modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}
}

func promotionDefinition(
	t *testing.T, modelID artifact.ID, residency recipe.ResidencyPolicy,
) recipe.Definition {
	t.Helper()
	definition, err := modelrecipe.InferenceWithModelDefinition(
		modelID, testutil.ArtifactID(t, artifact.KindProfile, "promotion-profile"),
		testutil.ArtifactID(t, artifact.KindModelDefinition, "promotion-definition"),
		recipe.PlacementHost, modelrecipe.DecodeSessionRequest, residency,
	)
	if err != nil {
		t.Fatal(err)
	}
	return definition
}

func publishRolloutReading(
	t *testing.T, store artifact.Repository, planID artifact.ID,
) evaluation.RolloutProjection {
	t.Helper()
	reading, err := evaluation.ProjectRolloutEvidence(context.Background(), store, planID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := evaluation.PublishRolloutProjection(context.Background(), store, reading); err != nil {
		t.Fatal(err)
	}
	return reading
}

// TestPromotionRequiresRolloutEvidenceClosure pins the promotion door:
// universal activation happens only after the complete closure — rollout
// plan, baseline reading, boundary reading strictly after it, observation
// coverage meeting the plan's contract, a committed rollback contract, and
// the live active recipe being the plan's own baseline — and the alias
// transition runs through the modelrecipe promotion owner. Insufficient
// coverage, readings bound to another plan, and a baseline that is not in
// service each refuse with the exact gap named.
func TestPromotionRequiresRolloutEvidenceClosure(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "promotion-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "rollout/promotion-fixture/facts",
		Artifacts: []artifact.Descriptor{
			{ID: modelID, Size: 4096},
			{ID: testutil.ArtifactID(t, artifact.KindProfile, "promotion-profile")},
			{ID: testutil.ArtifactID(t, artifact.KindModelDefinition, "promotion-definition")},
		},
	}); err != nil {
		t.Fatal(err)
	}
	baselineDefinition := promotionDefinition(t, modelID, recipe.ResidencyHostReference)
	candidateDefinition := promotionDefinition(t, modelID, recipe.ResidencyHostCache)
	if _, _, err := modelrecipe.PublishCandidate(
		ctx, store, "rollout/promotion-fixture/baseline-candidate", baselineDefinition,
	); err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, baselineDefinition,
		promotionVerification(t, store, baselineDefinition.ID, "rollout/promotion-fixture/baseline-verification"),
		recipe.EvidenceVerified, "baseline in service",
	); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(
		ctx, store, "rollout/promotion-fixture/candidate-candidate", candidateDefinition,
	); err != nil {
		t.Fatal(err)
	}

	authorities := map[string]artifact.ID{
		"admission":   testutil.ArtifactID(t, artifact.KindEvidence, "promotion-counterfactual"),
		"cohort":      testutil.ArtifactID(t, artifact.KindEvidence, "promotion-cohort-derivation"),
		"observation": testutil.ArtifactID(t, artifact.KindEvidence, "promotion-observation-derivation"),
		"rollback":    testutil.ArtifactID(t, artifact.KindEvidence, "promotion-rollback-decision"),
	}
	descriptors := make([]artifact.Descriptor, 0, len(authorities))
	for _, id := range authorities {
		descriptors = append(descriptors, artifact.Descriptor{ID: id})
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "rollout/promotion-fixture/authorities", Artifacts: descriptors,
	}); err != nil {
		t.Fatal(err)
	}
	plan, err := runrecord.NewRolloutPlan(runrecord.RolloutPlan{
		Baseline: baselineDefinition.ID, Candidate: candidateDefinition.ID,
		Admission: authorities["admission"], CohortKey: "request-digest",
		HashVersion: runrecord.RolloutHashVersion, CohortShare: 500,
		CohortEvidence: authorities["cohort"],
		Observation: runrecord.RolloutObservation{
			Metric: "exact-match", MinObservations: 2, Evidence: authorities["observation"],
		},
		Rollback: authorities["rollback"],
	})
	if err != nil {
		t.Fatal(err)
	}
	planBatch, err := plan.Batch("rollout/promotion-fixture/plan")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, planBatch); err != nil {
		t.Fatal(err)
	}
	baselineReading := publishRolloutReading(t, store, plan.ID)

	verification := promotionVerification(
		t, store, candidateDefinition.ID, "rollout/promotion-fixture/candidate-verification",
	)
	closure := RolloutPromotionClosure{
		Plan: plan.ID, BaselineReading: baselineReading.ID,
	}
	starved := publishRolloutReading(t, store, plan.ID)
	closure.CandidateReading = starved.ID
	if err := PromoteRolloutCandidate(
		ctx, store, closure, verification, "premature promotion",
	); err == nil || !strings.Contains(err.Error(), "observation contract") {
		t.Fatalf("starved coverage promoted: %v", err)
	}

	for _, name := range []string{"observation-one", "observation-two"} {
		observation := testutil.ArtifactID(t, artifact.KindEvidence, name)
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:       "rollout/promotion-fixture/" + name,
			Artifacts: []artifact.Descriptor{{ID: observation}},
			Lineage: []artifact.Lineage{{
				Child: observation, Parent: plan.ID, Relation: artifact.RelationDependsOn,
			}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	boundaryReading := publishRolloutReading(t, store, plan.ID)
	closure.CandidateReading = boundaryReading.ID

	foreign := RolloutPromotionClosure{
		Plan: plan.ID, BaselineReading: authorities["admission"], CandidateReading: boundaryReading.ID,
	}
	if err := PromoteRolloutCandidate(
		ctx, store, foreign, verification, "foreign reading promotion",
	); err == nil {
		t.Fatal("closure with a non-reading baseline promoted")
	}

	if err := PromoteRolloutCandidate(
		ctx, store, closure, verification, "rollout boundary promotion",
	); err != nil {
		t.Fatal(err)
	}
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !active || activation.Definition.ID != candidateDefinition.ID {
		t.Fatalf("promoted active = (%s, %v, %v)", activation.Definition.ID, active, err)
	}
	status, published, err := modelrecipe.Status(ctx, store, baselineDefinition.ID)
	if err != nil || !published || status != recipe.StatusSuperseded {
		t.Fatalf("superseded baseline = (%s, %v, %v)", status, published, err)
	}

	if err := PromoteRolloutCandidate(
		ctx, store, closure, verification, "repeat promotion",
	); err == nil || !strings.Contains(err.Error(), "not the rollout baseline") {
		t.Fatalf("promotion without its baseline in service: %v", err)
	}
}
