package discovery

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/pathidentity"
	"overgo/internal/testutil"
)

// TestExecutionDirectoryPrefersTheMostSpecificLiveDirectory records models
// the way their registrations do: a composite at the parent it shares with
// a sibling repository and at its own directory inside it, a model at one
// directory, and a model whose own directory is gone. The capability runs
// from the composite's own directory, from the one directory, and from the
// parent once its own directory is no longer live.
func TestExecutionDirectoryPrefersTheMostSpecificLiveDirectory(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parent, err := pathidentity.Canonical(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	child, dead := filepath.Join(parent, "LiveEdit"), filepath.Join(parent, "Moved")
	for _, directory := range []string{child, filepath.Join(parent, "Wan2.1-T2V-1.3B")} {
		if err := os.Mkdir(directory, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	record := func(name string, directories ...string) artifact.ID {
		t.Helper()
		component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, []byte(name))
		manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
			Role: artifact.ComponentWeights, Name: "weights", Artifact: component,
		}})
		if err != nil {
			t.Fatal(err)
		}
		batch := artifact.Batch{
			Key:       "fixture/discovery/execution-directory/" + name,
			Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(name))}},
			Manifests: []artifact.Manifest{manifest},
		}
		for _, directory := range directories {
			batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: artifact.Location{
				Artifact: manifest.ID, Kind: artifact.LocationDirectory, Value: directory,
			}, Action: artifact.LocationAdd})
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return manifest.ID
	}
	for _, want := range []struct {
		name        string
		directories []string
		directory   string
	}{
		{"composite", []string{parent, child}, child},
		{"single", []string{parent}, parent},
		{"moved", []string{parent, dead}, parent},
	} {
		model := record(want.name, want.directories...)
		if directory, err := ExecutionDirectory(ctx, store, model); err != nil || directory != want.directory {
			t.Errorf("%s recorded at %v runs from %q (%v), want %q", want.name, want.directories, directory, err, want.directory)
		}
	}
	if directory, err := ExecutionDirectory(ctx, store, record("gone", dead)); err == nil {
		t.Errorf("a model recorded only at a dead directory runs from %q", directory)
	}
}
