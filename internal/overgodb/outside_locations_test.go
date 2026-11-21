package overgodb

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestTheStoreListsTheLocationsItHoldsOutsideItsRoots holds the store's listing of what lies outside the repository and
// the data roots (store-roots): an artifact whose only live copy is outside is listed with it, which removing that
// folder would lose; one whose outside copy is gone and whose bytes the store does not hold is listed as lost; one with
// a live copy inside the roots, one recorded only inside them, one whose bytes the store holds and a remote location
// are not listed.
func TestTheStoreListsTheLocationsItHoldsOutsideItsRoots(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	repository, elsewhere := t.TempDir(), t.TempDir()
	write := func(path string) string {
		testcheck.NoError(t, os.WriteFile(path, []byte("bytes"), 0o600))
		return path
	}
	inside, outside, alsoOutside := write(filepath.Join(repository, "kept.bin")), write(filepath.Join(elsewhere, "only.bin")), write(filepath.Join(elsewhere, "copy.bin"))
	gone := filepath.Join(elsewhere, "removed", "tensors.safetensors")
	place := func(name string, content bool, locations ...artifact.Location) artifact.ID {
		descriptor := fixtureDescriptor(t, artifact.KindFile, name)
		batch := artifact.Batch{Key: "fixture/outside/" + name, Artifacts: []artifact.Descriptor{descriptor}}
		if content {
			batch = artifact.Batch{Key: batch.Key, Contents: []artifact.Content{{Descriptor: descriptor, Data: []byte(name)}}}
		}
		for _, location := range locations {
			location.Artifact = descriptor.ID
			batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
		}
		testcheck.Must(store.Commit(t.Context(), batch))(t)
		return descriptor.ID
	}
	file := func(path string) artifact.Location {
		return artifact.Location{Kind: artifact.LocationFile, Value: path}
	}
	onlyOutside := place("only-outside", false, file(outside))
	lost := place("lost", false, file(gone))
	place("kept-inside", false, file(alsoOutside), file(inside))
	place("inside-only", false, file(inside))
	place("held-content", true, file(gone))
	place("remote", false, artifact.Location{Kind: artifact.LocationRemote, Value: "remote://lane/model"})
	listed := testcheck.Must(store.OutsideLocations(t.Context(), []string{repository}))(t)
	if len(listed) != 2 {
		t.Fatalf("the store lists %+v outside its roots; want the artifact live only outside and the lost one", listed)
	}
	for _, entry := range listed {
		switch entry.Artifact {
		case onlyOutside:
			if entry.Lost() || !slices.Equal(entry.Live, []string{outside}) {
				t.Errorf("the artifact live only outside lists as %+v; want its live copy", entry)
			}
		case lost:
			if !entry.Lost() || !slices.Equal(entry.Gone, []string{gone}) {
				t.Errorf("the lost artifact lists as %+v; want it lost, its gone location named", entry)
			}
		default:
			t.Errorf("the store lists %+v; want only the artifact live only outside and the lost one", entry)
		}
	}
}
