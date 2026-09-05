package discovery

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// projectorFixture publishes a model, a projector whose bytes sit at the
// returned path, and the verified projection activation binding them.
func projectorFixture(t *testing.T, store *overgodb.Store, suffix string) (artifact.ID, string) {
	t.Helper()
	ctx := t.Context()
	modelPayload, projectorPayload := []byte("model-weights-"+suffix), []byte("projector-weights-"+suffix)
	modelComponent := testutil.ArtifactBytesID(t, artifact.KindTensorSet, modelPayload)
	projectorComponent := testutil.ArtifactBytesID(t, artifact.KindTensorSet, projectorPayload)
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	projectorPath := filepath.Join(t.TempDir(), "mmproj.gguf")
	for path, payload := range map[string][]byte{modelPath: modelPayload, projectorPath: projectorPayload} {
		if err := os.WriteFile(path, payload, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	model, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: modelComponent,
	}})
	if err != nil {
		t.Fatal(err)
	}
	projector, err := artifact.NewManifest(artifact.KindProjector, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: projectorComponent,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/discovery/projector/" + suffix,
		Artifacts: []artifact.Descriptor{
			{ID: modelComponent, Size: uint64(len(modelPayload))},
			{ID: projectorComponent, Size: uint64(len(projectorPayload))},
		},
		Manifests: []artifact.Manifest{model, projector},
		Locations: []artifact.LocationEvent{
			{Location: artifact.Location{Artifact: modelComponent, Kind: artifact.LocationFile, Value: modelPath}, Action: artifact.LocationAdd},
			{Location: artifact.Location{Artifact: projectorComponent, Kind: artifact.LocationFile, Value: projectorPath}, Action: artifact.LocationAdd},
		},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.ProjectionDefinition(model.ID, projector.ID, artifact.ID{}, recipe.DataImage)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/projection-candidate/"+suffix, definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/projection-validated/"+suffix, definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "fixture/discovery/projection-evidence/"+suffix, definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/discovery/projection-active/"+suffix, definition, verification,
		recipe.EvidenceVerified, "discovery projector fixture", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	return model.ID, projectorPath
}

// TestActiveProjectorResolvesDeclaredBytes proves the server's projector
// resolution: the active projection recipe names the projector, the
// store's location names its bytes, and only recorded bytes are served.
func TestActiveProjectorResolvesDeclaredBytes(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID, projectorPath := projectorFixture(t, store, "bound")
	memo := LoadMemo(ctx, store)
	location, ok, err := ActiveProjector(ctx, store, modelID, memo)
	if err != nil || !ok || location != projectorPath {
		t.Fatalf("ActiveProjector = (%q, %v, %v), want %q", location, ok, err, projectorPath)
	}

	// Replaced bytes at the recorded path are not the recorded projector.
	if err := os.WriteFile(projectorPath, []byte("replacement"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := ActiveProjector(ctx, store, modelID, LoadMemo(ctx, store)); ok || err == nil || !strings.Contains(err.Error(), "no recorded bytes") {
		t.Fatalf("replaced projector bytes resolved: ok=%v err=%v", ok, err)
	}
}

func TestActiveProjectorAbsentWithoutActivation(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "text-only-model")
	location, ok, err := ActiveProjector(ctx, store, modelID, LoadMemo(ctx, store))
	if err != nil || ok || location != "" {
		t.Fatalf("ActiveProjector without activation = (%q, %v, %v)", location, ok, err)
	}
}
