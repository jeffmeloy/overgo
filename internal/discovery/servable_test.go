package discovery

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestServableInvalidActivationNamesModel(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	component := testutil.ArtifactID(t, artifact.KindTensorSet, "discovery-weights")
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: component,
	}})
	if err != nil {
		t.Fatal(err)
	}
	profile := testutil.ArtifactID(t, artifact.KindProfile, "discovery-profile")
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "discovery-definition")
	intent := testutil.ArtifactID(t, artifact.KindEvidence, "discovery-legacy-intent")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/discovery/facts",
		Artifacts: []artifact.Descriptor{
			{ID: component, Size: 1}, {ID: profile, Size: 1},
			{ID: definitionID, Size: 1}, {ID: intent, Size: 1},
		},
		Manifests: []artifact.Manifest{manifest},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		manifest.ID, profile, definitionID, recipe.PlacementHost,
		modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/candidate", definition); err != nil {
		t.Fatal(err)
	}
	_, validated, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/validated", definition, recipe.StatusValidated, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	active, err := recipe.NewLifecycleEvent(
		definition, recipe.StatusValidated, recipe.StatusActive, &validated.ID, nil, []artifact.ID{intent},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := active.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"fixture/discovery/active", []artifact.Content{content}, nil,
		[]artifact.AliasBinding{
			{Name: "recipe.status." + definition.ID.String(), Target: active.ID, Previous: &validated.ID},
			{Name: "recipe.active.inference." + manifest.ID.String(), Target: definition.ID},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := Servable(ctx, store, 10); err == nil || !strings.Contains(err.Error(), manifest.ID.String()) {
		t.Fatalf("invalid activation error = %v", err)
	}
}
