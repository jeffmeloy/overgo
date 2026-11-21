package overgodb

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestAStoreRefusesALocationOutsideItsRoots holds a store with location roots to keeping its local locations inside
// them (store-roots): a file or directory inside is recorded; one outside is refused, naming it, and nothing of its
// batch is committed; a remote location and a removal are recorded wherever they point; and a store with no roots
// records a location anywhere, as a test's own store does.
func TestAStoreRefusesALocationOutsideItsRoots(t *testing.T) {
	t.Parallel()
	storeRoot, repository, elsewhere := filepath.Join(t.TempDir(), "store"), t.TempDir(), t.TempDir()
	outside := filepath.Join(elsewhere, "tmp", "tensors.safetensors")
	var store *Store
	located := func(name string, kind artifact.LocationKind, value string, action artifact.LocationAction) error {
		descriptor := fixtureDescriptor(t, artifact.KindFile, name)
		_, err := store.Commit(t.Context(), artifact.Batch{
			Key: fmt.Sprintf("fixture/roots/%s/%d", name, action), Artifacts: []artifact.Descriptor{descriptor},
			Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: descriptor.ID, Kind: kind, Value: value}, Action: action}},
		})
		return err
	}
	// Recorded before the store had roots, as the store's locations outside them were.
	store = testcheck.Must(Create(storeRoot))(t)
	testcheck.NoError(t, located("removed", artifact.LocationFile, outside, artifact.LocationAdd))
	testcheck.NoError(t, store.Close())
	testcheck.NoError(t, WriteLocationRoots(storeRoot, []string{repository}))
	store = testcheck.Must(Open(storeRoot))(t)
	defer store.Close()
	if err := located("inside", artifact.LocationFile, filepath.Join(repository, "testdata", "capture.bin"), artifact.LocationAdd); err != nil {
		t.Errorf("a file inside the roots is refused: %v", err)
	}
	if err := located("inside-directory", artifact.LocationDirectory, filepath.Join(repository, "models"), artifact.LocationAdd); err != nil {
		t.Errorf("a directory inside the roots is refused: %v", err)
	}
	if err := located("outside", artifact.LocationFile, outside, artifact.LocationAdd); err == nil || !strings.Contains(err.Error(), outside) {
		t.Errorf("a file outside the roots gives %v; want it refused, named", err)
	}
	_, found, err := store.Artifact(t.Context(), fixtureDescriptor(t, artifact.KindFile, "outside").ID)
	testcheck.NoError(t, err)
	if found {
		t.Error("the refused batch's artifact was committed")
	}
	if err := located("remote", artifact.LocationRemote, "remote://lane/model", artifact.LocationAdd); err != nil {
		t.Errorf("a remote location is refused: %v", err)
	}
	if err := located("removed", artifact.LocationFile, outside, artifact.LocationRemove); err != nil {
		t.Errorf("a removal of an outside location is refused: %v", err)
	}
	open := testcheck.Must(Create(filepath.Join(t.TempDir(), "open")))(t)
	defer open.Close()
	descriptor := fixtureDescriptor(t, artifact.KindFile, "anywhere")
	if _, err := open.Commit(t.Context(), artifact.Batch{
		Key: "fixture/roots/anywhere", Artifacts: []artifact.Descriptor{descriptor},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: descriptor.ID, Kind: artifact.LocationFile, Value: outside}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Errorf("a store with no roots refuses a location: %v", err)
	}
}
