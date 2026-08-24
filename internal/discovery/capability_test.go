package discovery

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestCapabilityCatalogListsNonInferenceActivations closes the "why only
// five servable" finding: a model whose only activation is a capability
// task (forecast here) is invisible to the inference-only Servable query
// but must appear in the capability catalog beside chat models, carrying
// its task and tier.
func TestCapabilityCatalogListsNonInferenceActivations(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	payload := []byte("capability-catalog-weights")
	component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
	location := filepath.Join(t.TempDir(), "forecaster.bin")
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
		Key:       "fixture/discovery/capability/facts",
		Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(payload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{
			Artifact: component, Kind: artifact.LocationFile, Value: location,
		}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskForecast, manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/capability/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/capability/validated", definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, "fixture/discovery/capability/evidence", definition.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/discovery/capability/active", definition, verification,
		recipe.EvidenceExperimental, "capability catalog fixture activation", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	servable, err := Servable(ctx, store, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(servable) != 0 {
		t.Fatalf("servable = %+v, want the forecast-only model invisible to the inference slice", servable)
	}
	entries, err := CapabilityCatalog(ctx, store, 100, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Model != manifest.ID {
		t.Fatalf("entries = %+v, want exactly the forecast model", entries)
	}
	entry := entries[0]
	if !entry.Present || entry.Location != location {
		t.Fatalf("entry = %+v, want recorded bytes present at the fixture location", entry)
	}
	if len(entry.Capabilities) != 1 {
		t.Fatalf("capabilities = %+v, want the single forecast activation", entry.Capabilities)
	}
	capability := entry.Capabilities[0]
	if capability.Task != recipe.TaskForecast || capability.Recipe != definition.ID ||
		capability.Tier != recipe.EvidenceExperimental || capability.Stale != "" {
		t.Fatalf("capability = %+v, want a trusted forecast activation at its activated tier", capability)
	}
}
