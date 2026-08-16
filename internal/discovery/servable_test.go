package discovery

import (
	"context"
	"os"
	"path/filepath"
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
	payload := []byte("discovery-weights")
	component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
	location := filepath.Join(t.TempDir(), "weights.gguf")
	if err := os.WriteFile(location, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: component,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/facts",
		Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(payload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{
			Artifact: component, Kind: artifact.LocationFile, Value: location,
		}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	publishLegacyActivation(t, store, manifest.ID, "present")
	if _, err := Servable(ctx, store, 10); err == nil || !strings.Contains(err.Error(), manifest.ID.String()) {
		t.Fatalf("invalid activation error = %v", err)
	}
}

func TestServableSkipsReplacedArtifactActivation(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	oldPayload, currentPayload := []byte("old-weights"), []byte("current-weights")
	oldComponent := testutil.ArtifactBytesID(t, artifact.KindTensorSet, oldPayload)
	location := filepath.Join(t.TempDir(), "weights.gguf")
	if err := os.WriteFile(location, currentPayload, 0o600); err != nil {
		t.Fatal(err)
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: oldComponent,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/replaced",
		Artifacts: []artifact.Descriptor{{ID: oldComponent, Size: uint64(len(oldPayload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{
			Artifact: oldComponent, Kind: artifact.LocationFile, Value: location,
		}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	publishLegacyActivation(t, store, manifest.ID, "replaced")
	entries, err := Servable(ctx, store, 10)
	if err != nil || len(entries) != 0 {
		t.Fatalf("servable replaced artifact = (%+v, %v)", entries, err)
	}
}

func publishLegacyActivation(t *testing.T, store artifact.Repository, modelID artifact.ID, suffix string) {
	t.Helper()
	ctx := context.Background()
	profile := testutil.ArtifactID(t, artifact.KindProfile, "discovery-profile-"+suffix)
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "discovery-definition-"+suffix)
	intent := testutil.ArtifactID(t, artifact.KindEvidence, "discovery-legacy-intent-"+suffix)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/legacy-facts/" + suffix,
		Artifacts: []artifact.Descriptor{{ID: profile}, {ID: definitionID}, {ID: intent}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		modelID, profile, definitionID, recipe.PlacementHost,
		modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/legacy-candidate/"+suffix, definition); err != nil {
		t.Fatal(err)
	}
	_, validated, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/legacy-validated/"+suffix, definition, recipe.StatusValidated, nil, nil,
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
		"fixture/discovery/legacy-active/"+suffix, []artifact.Content{content}, nil,
		[]artifact.AliasBinding{
			{Name: "recipe.status." + definition.ID.String(), Target: active.ID, Previous: &validated.ID},
			{Name: "recipe.active.inference." + modelID.String(), Target: definition.ID},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
}
