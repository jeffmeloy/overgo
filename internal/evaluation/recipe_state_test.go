package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAHeldOutReadingCarriesTheStateOfTheRecipeThatRan holds a held-out reading to its own recipe's state (SQA-283): the
// campaign read the model's inference state whatever task it ran, so an active generation model with no inference
// alias published its readings not verified, and a candidate recipe took the active one's state. The state a campaign
// publishes under is its recipe's: an active generation recipe reads verified, and a candidate recipe beside it, of
// another task nothing activated, reads not verified.
func TestAHeldOutReadingCarriesTheStateOfTheRecipeThatRan(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := newCapabilityEvaluationStore(t)
	weights := []byte("recipe-state-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest := testcheck.Must(artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}}))(t)
	testcheck.Must(store.Commit(ctx, artifact.Batch{Key: "state/model", Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Manifests: []artifact.Manifest{manifest}}))(t)
	active := testcheck.Must(modelrecipe.CapabilityDefinition(recipe.TaskGeneration, manifest.ID))(t)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "state/candidate/active", active); err != nil {
		t.Fatal(err)
	}
	verification := testcheck.Must(modelrecipetest.PublishVerification(ctx, store, "state/verification", active.ID))(t)
	testcheck.NoError(t, modelrecipe.ActivateCapability(ctx, store, active, verification, recipe.EvidenceParity, "recipe state fixture"))
	candidate := testcheck.Must(modelrecipe.CapabilityDefinition(recipe.TaskForecast, manifest.ID))(t)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "state/candidate/other", candidate); err != nil {
		t.Fatal(err)
	}

	if state := testcheck.Must(modelrecipe.ReadRecipeState(ctx, store, manifest.ID, active.ID))(t); state.Verification != modelrecipe.StateVerified {
		t.Fatalf("the active generation recipe reads %+v, want verified", state)
	}
	if state := testcheck.Must(modelrecipe.ReadRecipeState(ctx, store, manifest.ID, candidate.ID))(t); state.Verification != modelrecipe.StateNotVerified {
		t.Fatalf("a candidate recipe beside the active one reads %+v, want not verified", state)
	}
}
