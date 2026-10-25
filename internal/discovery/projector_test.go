package discovery

import (
	"context"
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

// componentBinding is a capability recipe that binds a component file to a model, and the resolver a launcher asks for
// the component's bytes: a projection's projector, a speculation's draft.
type componentBinding struct {
	name    string
	kind    artifact.Kind
	define  func(model, component artifact.ID) (recipe.Definition, error)
	resolve func(context.Context, *overgodb.Store, artifact.ID, *Memo) (string, bool, error)
}

var componentBindings = []componentBinding{
	{"projector", artifact.KindProjector, func(model, component artifact.ID) (recipe.Definition, error) {
		return modelrecipe.ProjectionDefinition(model, component, artifact.ID{}, recipe.DataImage)
	}, ActiveProjector},
	{"draft", artifact.KindDraft, modelrecipe.DraftDefinition, ActiveDraft},
}

// componentFixture publishes a model, a component whose bytes sit at the
// returned path, and the verified activation of the binding's recipe.
func componentFixture(t *testing.T, store *overgodb.Store, binding componentBinding, suffix string) (artifact.ID, string) {
	t.Helper()
	ctx := t.Context()
	modelPayload, componentPayload := []byte("model-weights-"+suffix), []byte(binding.name+"-weights-"+suffix)
	modelComponent := testutil.ArtifactBytesID(t, artifact.KindTensorSet, modelPayload)
	componentWeights := testutil.ArtifactBytesID(t, artifact.KindTensorSet, componentPayload)
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	componentPath := filepath.Join(t.TempDir(), binding.name+".gguf")
	for path, payload := range map[string][]byte{modelPath: modelPayload, componentPath: componentPayload} {
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
	component, err := artifact.NewManifest(binding.kind, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: componentWeights,
	}})
	if err != nil {
		t.Fatal(err)
	}
	key := "fixture/discovery/" + binding.name + "/" + suffix
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: key,
		Artifacts: []artifact.Descriptor{
			{ID: modelComponent, Size: uint64(len(modelPayload))},
			{ID: componentWeights, Size: uint64(len(componentPayload))},
		},
		Manifests: []artifact.Manifest{model, component},
		Locations: []artifact.LocationEvent{
			{Location: artifact.Location{Artifact: modelComponent, Kind: artifact.LocationFile, Value: modelPath}, Action: artifact.LocationAdd},
			{Location: artifact.Location{Artifact: componentWeights, Kind: artifact.LocationFile, Value: componentPath}, Action: artifact.LocationAdd},
		},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := binding.define(model.ID, component.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, key+"/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(ctx, store, key+"/validated", definition, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, key+"/evidence", definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, key+"/active", definition, verification, recipe.EvidenceVerified, "discovery "+binding.name+" fixture", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	return model.ID, componentPath
}

// TestActiveComponentsResolveDeclaredBytes proves the server's projector and
// draft resolution: the model's active projection or speculation recipe
// names the component, the store's location names its bytes, and only
// recorded bytes are served.
func TestActiveComponentsResolveDeclaredBytes(t *testing.T) {
	for _, binding := range componentBindings {
		t.Run(binding.name, func(t *testing.T) {
			ctx := t.Context()
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			modelID, componentPath := componentFixture(t, store, binding, "bound")
			location, ok, err := binding.resolve(ctx, store, modelID, LoadMemo(ctx, store))
			if err != nil || !ok || location != componentPath {
				t.Fatalf("resolution = (%q, %v, %v), want %q", location, ok, err, componentPath)
			}
			// Replaced bytes at the recorded path are not the recorded component.
			if err := os.WriteFile(componentPath, []byte("replacement"), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, ok, err := binding.resolve(ctx, store, modelID, LoadMemo(ctx, store)); ok || err == nil || !strings.Contains(err.Error(), "no recorded bytes") {
				t.Fatalf("replaced %s bytes resolved: ok=%v err=%v", binding.name, ok, err)
			}
		})
	}
}

func TestActiveComponentsAbsentWithoutActivation(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "text-only-model")
	for _, binding := range componentBindings {
		location, ok, err := binding.resolve(ctx, store, modelID, LoadMemo(ctx, store))
		if err != nil || ok || location != "" {
			t.Fatalf("%s without activation = (%q, %v, %v)", binding.name, location, ok, err)
		}
	}
}
