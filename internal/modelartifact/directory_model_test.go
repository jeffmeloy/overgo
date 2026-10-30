package modelartifact

import (
	"bytes"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

// TestADirectoryModelIsItsViewsIdentity holds a model registered from its own files to the identity its conversion's
// bytes would have: a view of a directory (gguf.View) is one component, the view's image identified as a written
// conversion's bytes are, located at the directory with its manifest. A claim the directory's memo makes stands in for
// the hash until the bytes are read, as a file's does, and the image's hash proves a true claim and refuses another.
func TestADirectoryModelIsItsViewsIdentity(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	stored := []byte("eight by two values of the directory's weights, read in place....")
	file, err := gguf.View([]gguf.Metadata{gguf.StringMetadata("general.architecture", "fixture")},
		[]gguf.ViewTensor{{Name: "weights", Shape: []uint64{uint64(len(stored))}, Type: gguf.DTypeI8, Payload: bytes.NewReader(stored)}},
		gguf.WriteOptions{}, directory)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	shard := file.Shards()[0]
	image := make([]byte, shard.Size)
	if _, err := shard.Source.ReadAt(image, 0); err != nil {
		t.Fatal(err)
	}
	want, _, err := artifact.Identify(artifact.KindTensorSet, bytes.NewReader(image))
	if err != nil {
		t.Fatal(err)
	}
	inventory, claims, err := FromGGUFClaimed(file, artifact.KindModel, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Components) != 1 || inventory.Components[0].ID != want || inventory.Components[0].Size != shard.Size ||
		len(inventory.Manifest.Components) != 1 || inventory.Manifest.Components[0].Artifact != want || claims != nil {
		t.Fatalf("the directory's inventory = %+v with claims %v, want its view's image %s as its one hashed component", inventory, claims, want)
	}
	for _, location := range inventory.Locations {
		if location.Kind != artifact.LocationDirectory || !strings.EqualFold(filepath.Clean(location.Value), filepath.Clean(directory)) {
			t.Errorf("a location of the directory's model is %+v, want the directory %s", location, directory)
		}
	}
	claimer := func(path string) (artifact.ID, uint64, bool) { return want, shard.Size, path == directory }
	if _, claimed, err := FromGGUFClaimed(file, artifact.KindModel, claimer); err != nil || len(claimed) != 1 || claimed[0] != want {
		t.Errorf("the claimed inventory's claims = %v, %v; want the remembered identity", claimed, err)
	}
	if err := VerifyClaims(file, []artifact.ID{want}); err != nil {
		t.Errorf("the view's true claim was refused: %v", err)
	}
	other := testutil.ArtifactID(t, artifact.KindTensorSet, "another image")
	if err := VerifyClaims(file, []artifact.ID{other}); !errors.Is(err, artifact.ErrContentMismatch) {
		t.Errorf("another claim on the view = %v, want a content mismatch", err)
	}
}
