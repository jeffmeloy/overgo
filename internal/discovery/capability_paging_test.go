package discovery

import (
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestCapabilityCatalogReadsEveryActivation pins the alias paging: a store
// holding more recipe artifacts than one query page still reports every
// activation, and the entry bound alone decides truncation.
func TestCapabilityCatalogReadsEveryActivation(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const bound = 64
	descriptors := make([]artifact.Descriptor, 0, 4*bound)
	for index := range 4 * bound {
		descriptors = append(descriptors, artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindRecipe, "paged-recipe-"+strconv.Itoa(index))})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/discovery/paged-recipes", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	weights := testutil.ArtifactBytesID(t, artifact.KindTensorSet, []byte("paged-weights"))
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weights,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/paged-model",
		Artifacts: []artifact.Descriptor{{ID: weights, Size: uint64(len("paged-weights"))}},
		Manifests: []artifact.Manifest{manifest},
	}); err != nil {
		t.Fatal(err)
	}
	modelID := manifest.ID
	publishVerifiedActivation(t, store, modelID, "paged")

	entries, truncated, err := CapabilityCatalog(ctx, store, bound, LoadMemo(ctx, store))
	if err != nil {
		t.Fatal(err)
	}
	if truncated {
		t.Fatal("one activated model reported as a truncated catalog")
	}
	for _, entry := range entries {
		if entry.Model != modelID {
			continue
		}
		for _, capability := range entry.Capabilities {
			if capability.Task == recipe.TaskInference && capability.Stale == "" {
				return
			}
		}
		t.Fatalf("catalog entry lacks the inference activation: %+v", entry.Capabilities)
	}
	t.Fatalf("catalog of %d entries omits the activated model behind %d recipe artifacts", len(entries), len(descriptors))
}
