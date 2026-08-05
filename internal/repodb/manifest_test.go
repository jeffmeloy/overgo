package repodb

import (
	"context"
	"testing"

	"llamacpp2go/internal/artifact"
)

const fixtureManifestLocation = "C:/models/fixture.gguf"

func TestManifestAndLocationReplay(t *testing.T) {
	root := t.TempDir()
	component := fixtureDescriptor(t, artifact.KindTensorSet, "manifest-weights")
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: component.ID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	batch := artifact.Batch{
		Key: "fixture/manifest/v1", Artifacts: []artifact.Descriptor{component},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{
			Location: artifact.Location{Artifact: component.ID, Kind: artifact.LocationFile, Value: fixtureManifestLocation},
			Action:   artifact.LocationAdd,
		}},
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	got, ok, err := store.Manifest(context.Background(), manifest.ID)
	if err != nil || !ok || !sameManifest(got, manifest) {
		t.Fatalf("manifest = (%+v, %v, %v)", got, ok, err)
	}
	parents, err := store.Parents(context.Background(), manifest.ID)
	if err != nil || len(parents) != 1 || parents[0].Parent != component.ID {
		t.Fatalf("manifest parents = (%+v, %v)", parents, err)
	}
	locations, err := store.Locations(context.Background(), component.ID)
	if err != nil || len(locations) != 1 || locations[0].Value != fixtureManifestLocation {
		t.Fatalf("locations = (%+v, %v)", locations, err)
	}
}

func TestLocationRemoval(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	descriptor := fixtureDescriptor(t, artifact.KindFile, "located")
	location := artifact.Location{Artifact: descriptor.ID, Kind: artifact.LocationFile, Value: fixtureManifestLocation}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/location/add", Artifacts: []artifact.Descriptor{descriptor},
		Locations: []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "fixture/location/remove",
		Locations: []artifact.LocationEvent{{Location: location, Action: artifact.LocationRemove}},
	}); err != nil {
		t.Fatal(err)
	}
	locations, err := store.Locations(context.Background(), descriptor.ID)
	if err != nil || len(locations) != 0 {
		t.Fatalf("locations after removal = (%+v, %v)", locations, err)
	}
}
