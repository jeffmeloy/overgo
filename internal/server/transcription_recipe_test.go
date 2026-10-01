package server

import (
	"context"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/speechrecognition"
)

// TestCompiledTranscriptionRecipeLoads holds a recipe the transcription
// compiler produces to loading wherever transcription runs: the fixture's
// small encoder compiles through modelrecipe.CompileTranscription, the
// server's transcription workspace serves it once active, the speech
// runtime's topology check accepts it, evaluate publishes its task model
// definition, and the native session evaluate transcribes through loads it.
func TestCompiledTranscriptionRecipeLoads(t *testing.T) {
	t.Parallel()
	fixture := newTranscriptionHTTPFixture(t, nil)
	ctx := t.Context()
	policy := fixture.workspace.policy
	definition, err := recipe.RequireDefinition(ctx, fixture.store, policy.Recipe)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.SpeechComponents(definition); err != nil {
		t.Fatalf("the speech runtime refuses the compiled recipe: %v", err)
	}
	if _, err := modelrecipe.PublishTaskModelDefinition(ctx, fixture.store, definition.ID); err != nil {
		t.Fatalf("evaluate cannot publish the recipe's task model definition: %v", err)
	}
	session, err := speechrecognition.LoadSession(ctx, fixture.store, definition.ID, policy.MemoryBytes)
	if err != nil {
		t.Fatalf("evaluate's native session cannot load the compiled recipe: %v", err)
	}
	if err := session.Close(context.WithoutCancel(ctx)); err != nil {
		t.Fatal(err)
	}
}
