package composition

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestCompositionPromotionActivation(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	ctx := t.Context()
	policy, err := LoadRepresentationBridgePromotionPolicy(ctx, store, authority.Recipe.PromotionPolicy)
	if err != nil || policy != authority.PromotionPolicy || authority.Promotion.PolicyID != policy.ID {
		t.Fatalf("recipe promotion policy = %+v, %v", policy, err)
	}
	batch, err := authority.Recipe.ActivationBatch(ctx, store, "fixture/composition/promoted-activate", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	active, found, err := ActiveComposition(
		ctx, store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
	)
	if err != nil || !found || active.ID != authority.Recipe.ID {
		t.Fatalf("promoted active composition = %+v, %t, %v", active, found, err)
	}
}

func TestUnpromotedBridgeRefused(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	ctx := t.Context()
	candidate := authority.Recipe
	candidate.Promotion = testutil.ArtifactID(t, artifact.KindEvidence, "missing bridge promotion")
	candidate, err := NewCompositionRecipe(candidate)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := candidate.Batch("fixture/composition/unpromoted-recipe")
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: candidate.Promotion})
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := candidate.ActivationBatch(ctx, store, "fixture/composition/unpromoted-activate", nil); err == nil {
		t.Fatal("recipe without exact promotion evidence activated")
	}
	candidate = authority.Recipe
	candidate.PromotionPolicy = testutil.ArtifactID(t, artifact.KindProfile, "unowned promotion policy")
	candidate, err = NewCompositionRecipe(candidate)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/composition/unowned-policy",
		Artifacts: []artifact.Descriptor{{ID: candidate.PromotionPolicy}},
	}); err != nil {
		t.Fatal(err)
	}
	batch, err = candidate.Batch("fixture/composition/unowned-policy-recipe")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := candidate.ActivationBatch(ctx, store, "fixture/composition/unowned-policy-activate", nil); err == nil {
		t.Fatal("recipe with descriptor-only promotion policy activated")
	}
}
