package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestAPolicyEditLeavesPublishedRecipesServable: a catalog edit refuses nothing already published: a recipe activates under
// the policy it holds, the one its verification read and its serving reads. MoveRuntimePolicy takes it to the catalog's and
// records why, and asks for a reason only when it moves anything.
func TestAPolicyEditLeavesPublishedRecipesServable(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := openProgramStore(t)
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "published before the edit")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/policy-edit/model", Artifacts: []artifact.Descriptor{{ID: model, Size: 4096}}}); err != nil {
		t.Fatal(err)
	}
	definition, err := inferenceFixture(model, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/policy-edit/candidate", definition); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := CatalogRuntimePolicy(definition.Task)
	if err != nil {
		t.Fatal(err)
	}
	earlier := catalog
	earlier.Interactive.OutputTokens++
	if earlier, err = runtimePolicyCodec.New(earlier); err != nil {
		t.Fatal(err)
	}
	content, err := runtimePolicyCodec.Content(earlier)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("fixture/policy-edit/earlier", []artifact.Content{content}, nil, []artifact.AliasBinding{{
		Name: runtimePolicyAlias(definition.ID), Target: earlier.ID, Previous: &catalog.ID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	held := func(want artifact.ID) {
		t.Helper()
		policy, err := BoundRuntimePolicy(ctx, store, definition)
		if err != nil || policy.ID != want {
			t.Fatalf("the recipe holds %v (%v), want %v", policy.ID, err, want)
		}
	}
	if err := ActivateCapability(
		ctx, store, definition, publishVerification(t, store, definition.ID, "fixture/policy-edit/verification"),
		recipe.EvidenceVerified, "a recipe published before the edit",
	); err != nil {
		t.Fatal(err)
	}
	held(earlier.ID)
	if err := MoveRuntimePolicy(ctx, store, definition, " "); err == nil {
		t.Fatal("a policy moved with no reason")
	}
	held(earlier.ID)
	const reason = "the catalog's default rose"
	if err := MoveRuntimePolicy(ctx, store, definition, reason); err != nil {
		t.Fatal(err)
	}
	held(catalog.ID)
	move, err := runtimePolicyMoveCodec.NewInitial(runtimePolicyMove{Recipe: definition.ID, From: earlier.ID, To: catalog.ID, Reason: reason})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runtimePolicyMoveCodec.Require(ctx, store, move.ID); err != nil {
		t.Fatalf("the store records no reason for the move: %v", err)
	}
	if err := MoveRuntimePolicy(ctx, store, definition, ""); err != nil {
		t.Fatalf("a recipe that already holds the catalog's policy: %v", err)
	}
}
