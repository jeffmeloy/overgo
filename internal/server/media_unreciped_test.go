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

// TestAMediaModelWithNoRecipeSaysWhatItNeeds: a media model the store holds with no recipe at all is listed, not left off
// the page, with what it needs to run, and it cannot be run; the listing a page reads still passes its checks. One whose
// files no source profile names lists nothing.
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
	got := unrecipedMedia(discovery.CatalogEntry{Model: waiting, Present: true, Location: filepath.Join(image, "model.safetensors")}, image)
	if len(got) != 1 || got[0].Model != waiting || got[0].Task != recipe.TaskImageGen || got[0].Recipe.Valid() ||
		!strings.Contains(got[0].Refusal, "validate -onboard "+filepath.ToSlash(image)) {
		t.Fatalf("listed %+v, want the waiting image model, saying what it needs", got)
	}
	if err := validateWorkflowCapabilities(got); err != nil {
		t.Fatalf("the page's listing refuses a model it shows: %v", err)
	}
	if other := unrecipedMedia(discovery.CatalogEntry{Model: testutil.ArtifactID(t, artifact.KindModel, "text"), Present: true,
		Location: filepath.Join(text, "model.safetensors")}, text); len(other) != 0 {
		t.Fatalf("listed %+v for a model no source profile names", other)
	}
}
