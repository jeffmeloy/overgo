package server

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestResponsesIdentifiersSeedFromDurableState closes the restart
// collision finding: a fresh handler over a store already carrying
// resp_7 must issue identifiers past it, never resp_1 again.
func TestResponsesIdentifiersSeedFromDurableState(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "seed-recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "seed-model")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "seed/authorities", Artifacts: []artifact.Descriptor{{ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.PublishInteraction(context.Background(), store, runrecord.Interaction{
		Response: "resp_7", Recipe: recipeID, Model: modelID, Node: recipe.NodeID("respond"),
	}, []runrecord.InteractionMessage{{Role: "assistant", Content: "recorded"}}); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	if next := handler.nextID.Add(1); next <= 7 {
		t.Fatalf("next identifier = %d, want the counter seeded past the durable resp_7", next)
	}
}
