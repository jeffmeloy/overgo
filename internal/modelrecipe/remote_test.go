package modelrecipe

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestPublishRemoteModelDefinitionBindsRelayRecipe: a hosted model's
// definition binds model + provider profile + relay recipe (idempotent);
// the binding check accepts it for that execution and refuses another
// model; a local recipe is not a remote definition's subject.
func TestPublishRemoteModelDefinitionBindsRelayRecipe(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "hosted-model")
	providerID := testutil.ArtifactID(t, artifact.KindProfile, "hosted-provider")
	testutil.PublishArtifact(t, store, modelID)
	testutil.PublishArtifact(t, store, providerID)
	definition, err := RemoteInferenceDefinition(modelID, providerID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/remote/candidate", definition); err != nil {
		t.Fatal(err)
	}
	published, err := PublishRemoteModelDefinition(ctx, store, definition.ID)
	if err != nil || published.Kind() != artifact.KindModelDefinition {
		t.Fatalf("published = %s, %v", published, err)
	}
	if again, err := PublishRemoteModelDefinition(ctx, store, definition.ID); err != nil || again != published {
		t.Fatalf("second publication = %s, %v", again, err)
	}
	if err := RequireModelDefinitionBinding(ctx, store, published, modelID, definition.ID); err != nil {
		t.Fatalf("binding refused: %v", err)
	}
	other := testutil.ArtifactID(t, artifact.KindModel, "other-model")
	if err := RequireModelDefinitionBinding(ctx, store, published, other, definition.ID); err == nil || !strings.Contains(err.Error(), "differs") {
		t.Fatalf("binding to another model: %v", err)
	}
	local, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/remote/local-candidate", local); err != nil {
		t.Fatal(err)
	}
	if _, err := PublishRemoteModelDefinition(ctx, store, local.ID); err == nil || !strings.Contains(err.Error(), "not a remote relay") {
		t.Fatalf("local recipe as a remote definition: %v", err)
	}
}
