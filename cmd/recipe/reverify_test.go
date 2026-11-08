package main

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAnOutOfDateModelIsReverifiedOnItsRecordedSuite holds re-verification to the suite the store recorded: on
// 2026-11-08 five servable models, the judge among them, held evidence for an architecture profile their build no longer
// compiled. A model's accepted recipe was verified on two exact cases and is active; the recipe the build now derives is
// another. Re-verified through a runtime that answers wrong, it is refused and the accepted recipe stays active; through
// one that answers right, both recorded cases are asked again, the derived recipe is activated on what that records, and
// a model whose active recipe is the derived one has nothing to re-verify.
func TestAnOutOfDateModelIsReverifiedOnItsRecordedSuite(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := testcheck.Must(overgodb.Create(t.TempDir()))(t)
	t.Cleanup(func() { testcheck.NoError(t, store.Close()) })
	model := testutil.ArtifactID(t, artifact.KindModel, "reverified fixture")
	modelDefinition := testutil.ArtifactID(t, artifact.KindModelDefinition, "reverified model")
	checkpoint := testutil.ArtifactID(t, artifact.KindCheckpoint, "derived surface")
	for _, id := range []artifact.ID{model, modelDefinition, checkpoint} {
		testutil.PublishArtifact(t, store, id)
	}
	accepted := testcheck.Must(modelrecipe.CapabilityDefinition(recipe.TaskSpeech, model))(t)
	derived := testcheck.Must(modelrecipe.AdaptedSpeechDefinition(accepted, checkpoint))(t)
	for _, definition := range []recipe.Definition{accepted, derived} {
		if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/"+definition.ID.String(), definition); err != nil {
			t.Fatal(err)
		}
	}
	suite := evaluation.ExactSuite{Schema: "test/v1", Source: "accepted suite", Cases: []evaluation.ExactCase{
		{Name: "first", Prompt: "first", Text: "ok", MaxTokens: 1, PromptTokens: 1, GeneratedTokens: 1},
		{Name: "second", Prompt: "second", Text: "ok", MaxTokens: 1, PromptTokens: 1, GeneratedTokens: 1},
	}}
	exact := testcheck.Must(evaluation.CompileExact(suite))(t)
	answering := func(text string) (*deferredExactRuntime, *deferredExactFixture) {
		fixture := &deferredExactFixture{text: text}
		return &deferredExactRuntime{open: func(context.Context) (exactRuntime, error) { return fixture, nil }}, fixture
	}
	right, _ := answering("ok")
	verification := testcheck.Must(verifyExactRuntime(ctx, store, accepted, strings.Repeat("a", 40), suite, exact, modelDefinition, right))(t)
	testcheck.NoError(t, modelrecipe.ActivateCapability(ctx, store, accepted, verification, recipe.EvidenceVerified, "fixture"))
	activeRecipe := func() artifact.ID {
		t.Helper()
		active, found, err := modelrecipe.ActiveRecord(ctx, store, model, recipe.TaskSpeech)
		testcheck.NoError(t, err)
		if !found {
			t.Fatal("the model has no active recipe")
		}
		return active.Definition.ID
	}
	wrong, _ := answering("no")
	if _, err := reverifyOn(ctx, store, model, derived, strings.Repeat("b", 40), modelDefinition, wrong); err == nil || activeRecipe() != accepted.ID {
		t.Fatalf("a derived recipe that answers the recorded cases wrong was re-verified (%v), or the accepted recipe left", err)
	}
	right, asked := answering("ok")
	reverified := testcheck.Must(reverifyOn(ctx, store, model, derived, strings.Repeat("c", 40), modelDefinition, right))(t)
	testcheck.NoError(t, modelrecipe.ActivateCapability(ctx, store, derived, reverified, recipe.EvidenceVerified, reverificationReason))
	if asked.calls != len(suite.Cases) || activeRecipe() != derived.ID {
		t.Fatalf("re-verification asked %d of the %d recorded cases and left %s active, want the derived %s", asked.calls, len(suite.Cases), activeRecipe(), derived.ID)
	}
	again, _ := answering("ok")
	if _, err := reverifyOn(ctx, store, model, derived, strings.Repeat("d", 40), modelDefinition, again); err == nil || !strings.Contains(err.Error(), "nothing to re-verify") {
		t.Fatalf("a model whose active recipe is the derived one was re-verified: %v", err)
	}
}
