package server

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestBrowserJourneyPreparation requires preparation to retain
// its catalog hashes even for media models not selected for text serving.
func TestBrowserJourneyPreparation(t *testing.T) {
	for _, complete := range []bool{false, true} {
		t.Run(fmt.Sprintf("explicit_configuration_%t", complete), func(t *testing.T) {
			t.Setenv("OVERGO_DATA_ROOT", filepath.Join(t.TempDir(), "absent"))
			t.Setenv("OVERGO_WEBUI_LANE_SERVER", "declared-server")
			t.Setenv("OVERGO_WEBUI_LANE_STORE", "declared-store")
			t.Setenv("OVERGO_WEBUI_LANE_MODEL", "declared-model")
			t.Setenv("OVERGO_WEBUI_LANE_MODEL_LOCATION", "")
			t.Setenv("OVERGO_WEBUI_LANE_MULTIMODAL_MODEL", "declared-vision")
			if complete {
				t.Setenv("OVERGO_WEBUI_LANE_MODEL_LOCATION", "declared-location")
			}
			configuration, err := prepareBrowserJourney(t)
			if complete {
				if err != nil || configuration.binary != "declared-server" || configuration.store != "declared-store" || configuration.model != "declared-model" || configuration.location != "declared-location" || configuration.multimodal != "declared-vision" {
					t.Fatalf("explicit configuration changed: %+v %v", configuration, err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "incomplete explicit") {
				t.Fatalf("partial configuration mixed with automatic discovery: %+v %v", configuration, err)
			}
		})
	}
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	payload := []byte("lane fixture weights")
	weights := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: weights}})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "lane-fixture/model", Artifacts: []artifact.Descriptor{{ID: weights, Size: uint64(len(payload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: weights, Kind: artifact.LocationFile, Value: path}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskForecast, manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipetest.PublishActivation(ctx, store, "lane-fixture/active", definition); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if servable, multimodal, err := smallestServables(ctx, root); err != nil || len(servable) != 0 || len(multimodal) != 0 {
		t.Fatalf("non-inference selection = %v, %v, %v", servable, multimodal, err)
	}
	store, err = overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	before, bound, err := artifact.ResolveAlias(ctx, store, discovery.IdentityEvidenceAlias)
	if err != nil || !bound {
		t.Fatalf("preparation did not persist identities: %v, %v", bound, err)
	}
	memo := discovery.LoadMemo(ctx, store)
	entries, _, err := discovery.CapabilityCatalog(ctx, store, laneCatalogLimit, memo)
	if err != nil || len(entries) != 1 || !entries[0].Present {
		t.Fatalf("fresh reader = %+v, %v", entries, err)
	}
	if err := discovery.PublishMemo(ctx, store, memo); err != nil {
		t.Fatal(err)
	}
	after, _, err := artifact.ResolveAlias(ctx, store, discovery.IdentityEvidenceAlias)
	if err != nil || before != after {
		t.Fatalf("unchanged identities republished: %v -> %v, %v", before, after, err)
	}
	if err := os.WriteFile(path, []byte("changed lane fixture weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	entries, _, err = discovery.CapabilityCatalog(ctx, store, laneCatalogLimit, discovery.LoadMemo(ctx, store))
	if err != nil || len(entries) != 1 || entries[0].Present {
		t.Fatalf("changed model bytes admitted: %+v, %v", entries, err)
	}
}
