package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestCapabilityEvidenceSelector(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	weights := []byte("selector-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "selector/model",
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Manifests: []artifact.Manifest{manifest},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := CapabilityDefinition(recipe.TaskGeneration, manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "selector/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification := publishVerification(t, store, definition.ID, "selector/verification")
	if err := ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceParity, "selector fixture",
	); err != nil {
		t.Fatal(err)
	}
	const alias = "capability/generation/default"
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "selector/alias",
		Aliases: []artifact.AliasBinding{{Name: alias, Target: manifest.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	identities := make(map[artifact.ID]struct{})
	sessions := []SessionSelection{SessionPin, SessionWarm, SessionSpillover}
	for _, session := range sessions {
		selected, err := ResolveCapabilityEvidenceSelector(ctx, store, CapabilityEvidenceSelector{
			Alias: alias, Task: recipe.TaskGeneration, Session: session,
		})
		if err != nil {
			t.Fatal(err)
		}
		if selected.Activation.Definition.ID != definition.ID ||
			selected.Program.Definition().ID != definition.ID ||
			selected.Resources.Model != manifest.ID ||
			selected.Resources.ArtifactBytes != uint64(len(weights)) ||
			selected.Resources.Session != recipe.SessionCapacity {
			t.Fatalf("%s selection=%+v", session, selected)
		}
		identities[selected.Identity] = struct{}{}
	}
	if len(identities) != len(sessions) {
		t.Fatalf("selector identities=%d", len(identities))
	}
	if _, err := ResolveCapabilityEvidenceSelector(ctx, store, CapabilityEvidenceSelector{
		Alias: "capability/generation/missing", Task: recipe.TaskGeneration, Session: SessionWarm,
	}); err == nil {
		t.Fatal("missing model alias accepted")
	}
}
