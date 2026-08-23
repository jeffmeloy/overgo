package runrecord

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestInteractionIdentity(t *testing.T) {
	messages := []InteractionMessage{{Role: "user", Content: "hello"}}
	first, err := NewInteractionTranscript(messages)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewInteractionTranscript(messages)
	if err != nil || first.ID != second.ID {
		t.Fatalf("transcript identity = (%s, %s, %v)", first.ID, second.ID, err)
	}
	interaction, err := NewInteraction(Interaction{Response: "resp_fixture", Message: first.ID})
	if err != nil || interaction.ID.Kind() != artifact.KindEvidence {
		t.Fatalf("interaction identity = (%s, %v)", interaction.ID, err)
	}
}

func TestInteractionLineage(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "interaction-recipe")
	parentID := testutil.ArtifactID(t, artifact.KindEvidence, "parent-interaction")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "interaction/parents", Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: parentID}},
	}); err != nil {
		t.Fatal(err)
	}
	published, err := PublishInteraction(context.Background(), store, Interaction{
		Response: "resp_lineage", Recipe: recipeID, Parent: parentID,
	}, []InteractionMessage{{Role: "assistant", Content: "answer"}})
	if err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(context.Background(), published.ID)
	if err != nil || len(parents) != 3 {
		t.Fatalf("interaction parents = (%v, %v)", parents, err)
	}
	resolved, found, err := ResolveInteraction(context.Background(), store, published.Response)
	if err != nil || !found || resolved.ID != published.ID {
		t.Fatalf("resolved interaction = (%s, %v, %v)", resolved.ID, found, err)
	}
}
