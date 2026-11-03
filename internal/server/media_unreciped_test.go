package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestAMediaModelWithNoRecipeSaysWhatItNeeds: a media model the store holds with no active recipe is listed, not left off
// the page, with what it needs to run, and it cannot be run. A model already listed, one with an activation, one absent and
// one whose files no source profile names list nothing more.
func TestAMediaModelWithNoRecipeSaysWhatItNeeds(t *testing.T) {
	t.Parallel()
	directory := func(config string) string {
		path := t.TempDir()
		if err := os.WriteFile(filepath.Join(path, "config.json"), []byte(config), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	image, text := directory(`{"model_type":"uvit"}`), directory(`{"model_type":"llama"}`)
	waiting := testutil.ArtifactID(t, artifact.KindModel, "waiting")
	listedModel := testutil.ArtifactID(t, artifact.KindModel, "listed")
	registered := []discovery.CatalogEntry{
		{Model: waiting, Present: true, Location: filepath.Join(image, "model.safetensors")},
		{Model: listedModel, Present: true, Location: filepath.Join(image, "other.safetensors")},
		{Model: testutil.ArtifactID(t, artifact.KindModel, "active"), Present: true, Location: filepath.Join(image, "a.safetensors"),
			Capabilities: []discovery.Capability{{Task: recipe.TaskImageGen}}},
		{Model: testutil.ArtifactID(t, artifact.KindModel, "absent"), Location: filepath.Join(image, "gone.safetensors")},
		{Model: testutil.ArtifactID(t, artifact.KindModel, "text"), Present: true, Location: filepath.Join(text, "model.safetensors")},
	}
	got := unrecipedMedia(registered, map[artifact.ID]bool{listedModel: true}, func(entry discovery.CatalogEntry) (string, error) {
		return filepath.Dir(entry.Location), nil
	})
	if len(got) != 1 || got[0].Model != waiting || got[0].Task != recipe.TaskImageGen || got[0].Recipe.Valid() ||
		!strings.Contains(got[0].Refusal, "validate -onboard "+filepath.ToSlash(image)) {
		t.Fatalf("listed %+v, want the waiting image model alone, saying what it needs", got)
	}
}
