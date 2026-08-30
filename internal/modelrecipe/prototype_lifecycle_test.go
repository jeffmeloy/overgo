package modelrecipe

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func prototypeCandidate(
	t *testing.T, resolved ResolvedModelDefinition,
	session DecodeSessionPolicy, residency recipe.ResidencyPolicy,
) recipe.Definition {
	t.Helper()
	candidates, err := DerivePrototypeRecipes(
		resolved, recipe.PlacementHost, session, residency, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	return candidates[recipe.TaskInference]
}

func promoteCandidate(
	t *testing.T, store artifact.Repository, definition recipe.Definition, prefix string,
) {
	t.Helper()
	ctx := context.Background()
	closure := ValidateCandidateRecipe(definition)
	if !closure.Runnable {
		t.Fatalf("prototype candidate refused: %v", closure.Refusals)
	}
	if _, _, err := PublishCandidate(ctx, store, prefix+"/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, err := PublishCandidateClosure(ctx, store, closure); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(
		ctx, store, prefix+"/validated", definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
}

// TestPrototypeRecipeLifecycleAndRollback drives prototype-derived recipes
// through the whole lifecycle: candidate, validated, verified activation,
// refusal, supersession, retirement on failed proof, and rollback. Rollback
// is the load-bearing claim: after the successor retires, the predecessor
// returns to service only through a fresh verified activation bound to its
// own identity — the retired recipe stays superseded, a healthy activation
// refuses to roll back, and the rollback cites the retirement it answers.
func TestPrototypeRecipeLifecycleAndRollback(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	resolved := derivationResolvedDefinition(t)
	modelID := resolved.Document.Model
	testutil.PublishArtifact(t, store, modelID)
	testutil.PublishArtifact(t, store, resolved.Profile.ID)
	testutil.PublishArtifact(t, store, resolved.Document.ID)

	first := prototypeCandidate(t, resolved, DecodeSessionRequest, recipe.ResidencyHostReference)
	promoteCandidate(t, store, first, "fixture/prototype/first")
	firstVerification := publishVerification(t, store, first.ID, "fixture/prototype/first/verification")
	if _, _, err := ActivateVerified(
		ctx, store, "fixture/prototype/first/active", first, firstVerification,
		recipe.EvidenceVerified, "prototype activation", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	if err := RollbackActivation(
		ctx, store, modelID, recipe.TaskInference, firstVerification,
		recipe.EvidenceVerified, "premature rollback",
	); err == nil || !strings.Contains(err.Error(), "retire it with failed proof") {
		t.Fatalf("healthy activation rolled back: %v", err)
	}

	refused := prototypeCandidate(t, resolved, DecodeSessionCapacity, recipe.ResidencyHostCache)
	if _, _, err := PublishCandidate(ctx, store, "fixture/prototype/refused/candidate", refused); err != nil {
		t.Fatal(err)
	}
	refusal, err := recipe.NewDecision(
		refused.ID, recipe.DecisionRefused, recipe.EvidenceVerified, "prototype parity failed",
		recipe.Decider{CodeCommit: lifecycleDecisionCommit, Derivation: firstVerification.Gate},
		[]artifact.ID{firstVerification.Gate},
	)
	if err != nil {
		t.Fatal(err)
	}
	publishDecision(t, store, "fixture/prototype/refusal", refusal)
	if _, _, err := Transition(
		ctx, store, "fixture/prototype/refused", refused, recipe.StatusRefused,
		[]artifact.ID{refusal.ID}, nil,
	); err != nil {
		t.Fatal(err)
	}

	second := prototypeCandidate(t, resolved, DecodeSessionRequest, recipe.ResidencyHostCache)
	promoteCandidate(t, store, second, "fixture/prototype/second")
	secondVerification := publishVerification(t, store, second.ID, "fixture/prototype/second/verification")
	if _, _, err := ActivateVerified(
		ctx, store, "fixture/prototype/second/active", second, secondVerification,
		recipe.EvidenceVerified, "prototype supersession", nil, &first.ID,
	); err != nil {
		t.Fatal(err)
	}
	if status, published, err := Status(ctx, store, first.ID); err != nil || !published ||
		status != recipe.StatusSuperseded {
		t.Fatalf("superseded predecessor = (%s, %v, %v)", status, published, err)
	}

	failed := publishFailedVerification(t, store, second.ID, "fixture/prototype/failed-verification")
	if err := RetireActiveCapability(
		ctx, store, second, failed, "prototype regression on promotion evidence",
	); err != nil {
		t.Fatal(err)
	}
	if declared, err := HasActiveRecipe(ctx, store, modelID, recipe.TaskInference); err != nil || declared {
		t.Fatalf("retired activation still declared = (%v, %v)", declared, err)
	}

	rollbackVerification := publishVerification(t, store, first.ID, "fixture/prototype/rollback/verification")
	if err := RollbackActivation(
		ctx, store, modelID, recipe.TaskInference, secondVerification,
		recipe.EvidenceVerified, "rollback with mismatched proof",
	); err == nil {
		t.Fatal("rollback accepted verifier proof bound to the retired recipe")
	}
	if err := RollbackActivation(
		ctx, store, modelID, recipe.TaskInference, rollbackVerification,
		recipe.EvidenceVerified, "restore last verified predecessor",
	); err != nil {
		t.Fatal(err)
	}
	record, ok, err := ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok || record.Definition.ID != first.ID {
		t.Fatalf("rollback active record = (%s, %v, %v)", record.Definition.ID, ok, err)
	}
	if record.Event.Supersedes == nil || *record.Event.Supersedes != second.ID {
		t.Fatalf("rollback supersession = %+v", record.Event.Supersedes)
	}
	if status, published, err := Status(ctx, store, second.ID); err != nil || !published ||
		status != recipe.StatusSuperseded {
		t.Fatalf("retired recipe after rollback = (%s, %v, %v)", status, published, err)
	}
}
