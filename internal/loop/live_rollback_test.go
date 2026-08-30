package loop

import (
	"context"
	"encoding/json"
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

func promotionFailedVerification(
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
		runrecord.OutcomeFailed, "live-regression", 1,
		[]runrecord.GateStep{{
			Name: "verify", Phase: runrecord.PhaseValidate,
			Outcome: runrecord.StepFailed, DurationNS: 1,
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

func publishBreakerTransition(
	t *testing.T, store artifact.Repository, prior uint64,
) artifact.ID {
	t.Helper()
	ctx := context.Background()
	historyEvidence := testutil.ArtifactID(t, artifact.KindEvidence, "rollback-history")
	windowEvidence := testutil.ArtifactID(t, artifact.KindEvidence, "rollback-window-derivation")
	testutil.PublishArtifact(t, store, historyEvidence)
	testutil.PublishArtifact(t, store, windowEvidence)
	window, err := evaluation.DeriveLiveSafetyWindow(
		"exact-match", runrecord.DirectionMaximize,
		[]float64{0.52, 0.58, 0.55, 0.50, 0.60, 0.54, 0.53, 0.57, 0.51, 0.56},
		historyEvidence,
	)
	if err != nil {
		t.Fatal(err)
	}
	transition, err := JudgeCircuitBreaker(
		window, windowEvidence,
		breakerObservations(0.40, int(window.RequiredObservations)), prior,
	)
	if err != nil {
		t.Fatal(err)
	}
	published, err := PublishCircuitBreakerTransition(ctx, store, transition)
	if err != nil {
		t.Fatal(err)
	}
	return published
}

// TestAtomicCandidateRollback pins the rollback door: only a containment-
// level breaker transition can cause one, the decision binds the exact
// judged candidate so a stale decision refuses once the active alias has
// moved, the retire and reactivate compare-and-set transitions run through
// the modelrecipe promotion owner, and the published causal record links
// the regression evidence to the retired recipe and the restored
// predecessor — publishing only after the rollback holds.
func TestAtomicCandidateRollback(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "promotion-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "live-safety/rollback-fixture/facts",
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
			ctx, store, "live-safety/rollback-fixture/"+name, definition,
		); err != nil {
			t.Fatal(err)
		}
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, baseline,
		promotionVerification(t, store, baseline.ID, "live-safety/rollback-fixture/baseline-verification"),
		recipe.EvidenceVerified, "baseline in service",
	); err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, candidate,
		promotionVerification(t, store, candidate.ID, "live-safety/rollback-fixture/candidate-verification"),
		recipe.EvidenceVerified, "candidate promoted",
	); err != nil {
		t.Fatal(err)
	}

	quarantine := publishBreakerTransition(t, store, 2)
	advisory := publishBreakerTransition(t, store, 0)
	retirement := promotionFailedVerification(
		t, store, candidate.ID, "live-safety/rollback-fixture/failed-verification",
	)
	reactivation := promotionVerification(
		t, store, baseline.ID, "live-safety/rollback-fixture/reactivation-verification",
	)

	if _, err := RollbackQuarantinedCandidate(
		ctx, store, modelID, recipe.TaskInference, candidate.ID, advisory,
		retirement, reactivation, "advisory rollback",
	); err == nil || !strings.Contains(err.Error(), "not a containment transition") {
		t.Fatalf("advisory transition rolled back: %v", err)
	}
	if _, err := RollbackQuarantinedCandidate(
		ctx, store, modelID, recipe.TaskInference, baseline.ID, quarantine,
		retirement, reactivation, "stale rollback",
	); err == nil || !strings.Contains(err.Error(), "moved past the judged candidate") {
		t.Fatalf("stale decision rolled back: %v", err)
	}

	record, err := RollbackQuarantinedCandidate(
		ctx, store, modelID, recipe.TaskInference, candidate.ID, quarantine,
		retirement, reactivation, "sustained live regression",
	)
	if err != nil {
		t.Fatal(err)
	}
	restored, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !active || restored.Definition.ID != baseline.ID {
		t.Fatalf("restored active = (%s, %v, %v)", restored.Definition.ID, active, err)
	}
	status, published, err := modelrecipe.Status(ctx, store, candidate.ID)
	if err != nil || !published || status != recipe.StatusSuperseded {
		t.Fatalf("retired candidate = (%s, %v, %v)", status, published, err)
	}
	content, found, err := artifact.ReadContent(ctx, store, record)
	if err != nil || !found {
		t.Fatalf("causal record was not published: (%v, %v)", found, err)
	}
	var causal LiveRollbackRecord
	if err := json.Unmarshal(content.Data, &causal); err != nil {
		t.Fatal(err)
	}
	if causal.Retired != candidate.ID || causal.Restored != baseline.ID || causal.Transition != quarantine {
		t.Fatalf("causal record = %+v", causal)
	}
	parents, err := store.Parents(ctx, record)
	if err != nil {
		t.Fatal(err)
	}
	cited := map[artifact.ID]bool{}
	for _, edge := range parents {
		cited[edge.Parent] = true
	}
	if !cited[quarantine] || !cited[candidate.ID] || !cited[baseline.ID] {
		t.Fatalf("causal lineage lost a citation: %v", parents)
	}

	if _, err := RollbackQuarantinedCandidate(
		ctx, store, modelID, recipe.TaskInference, candidate.ID, quarantine,
		retirement, reactivation, "repeat rollback",
	); err == nil || !strings.Contains(err.Error(), "moved past the judged candidate") {
		t.Fatalf("repeated stale decision rolled back: %v", err)
	}
}
