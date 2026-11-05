package server

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheCatalogSaysWhatAnUnactivatedModelRuns: the catalog the chat page's chooser reads names, for a model nothing has
// activated, the tasks of the recipes its registration published, beside "not verified": a model whose candidate runs as
// text is one the chooser offers to serve, as a trained model is, and one whose candidate makes images is not. An
// activated model names none.
func TestTheCatalogSaysWhatAnUnactivatedModelRuns(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	weights := func(name string) string {
		location := filepath.Join(t.TempDir(), name)
		testcheck.NoError(t, os.WriteFile(location, []byte(name), 0o600))
		return location
	}
	text, _ := candidateModelAt(t, store, weights("trained.gguf"), func(model artifact.ID) (recipe.Definition, error) {
		profile := testutil.ArtifactID(t, artifact.KindProfile, "chooser-profile")
		definition := testutil.ArtifactID(t, artifact.KindModelDefinition, "chooser-definition")
		testutil.PublishArtifact(t, store, profile)
		testutil.PublishArtifact(t, store, definition)
		return modelrecipe.InferenceWithModelDefinition(model, profile, definition,
			recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference)
	})
	image, _ := candidateModelAt(t, store, weights("image.safetensors"), oscillatorImageDefinition)
	activated := activateModelAt(t, store, weights("activated.safetensors"), oscillatorImageDefinition)
	handler, err := New(Config{Repository: store}, &fakeGenerator{})
	testcheck.NoError(t, err)
	t.Cleanup(func() { _ = handler.Close() })
	response := serveTestRequest(handler, http.MethodGet, "/catalog/models", "")
	var catalog struct{ Models []catalogModel }
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &catalog) != nil {
		t.Fatalf("catalog: %d %s", response.Code, response.Body)
	}
	listed := map[string]catalogModel{}
	for _, model := range catalog.Models {
		listed[model.Model] = model
	}
	for model, want := range map[artifact.ID][]recipe.Task{text: {recipe.TaskInference}, image: {recipe.TaskImageGen}, activated: nil} {
		entry, found := listed[model.String()]
		if !found || !entry.Present || !slices.Equal(entry.Candidates, want) {
			t.Fatalf("model %s listed %+v, want present with candidates %v", model, entry, want)
		}
		if want != nil && (entry.Verification != string(modelrecipe.StateNotVerified) || entry.Recipe != "") {
			t.Fatalf("unactivated model %s listed %+v, want it not verified with no active recipe", model, entry)
		}
	}
}
