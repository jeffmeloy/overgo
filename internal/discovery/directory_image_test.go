package discovery

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestADirectoryThatIsItsConversionsImageServesFromItself holds the catalog to serving a model from its own files where
// its directory's view is its conversion's image: Carbon-500M's view is its conversion's to the byte, one tensor set
// recorded at the GGUF and at the directory, and the catalog served the GGUF. A directory holding the component's bytes
// is served from ahead of the file; a directory that does not hold them leaves the file served.
func TestADirectoryThatIsItsConversionsImageServesFromItself(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	root := t.TempDir()
	conversion, directory := filepath.Join(root, "Tiny-LM-f16.gguf"), filepath.Join(root, "Tiny-LM")
	component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, []byte("tiny-lm"))
	descriptor := artifact.Descriptor{ID: component, Size: 7}
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: component,
	}})
	testcheck.NoError(t, err)
	batch := artifact.Batch{Key: "fixture/discovery/directory-image", Artifacts: []artifact.Descriptor{descriptor}, Manifests: []artifact.Manifest{manifest}}
	for _, location := range []artifact.Location{
		{Artifact: component, Kind: artifact.LocationFile, Value: conversion},
		{Artifact: component, Kind: artifact.LocationDirectory, Value: directory},
	} {
		batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	held := fileIdentity{present: true, id: component, size: descriptor.Size}
	key := func(path string) string { return path + "\x00" + artifact.KindTensorSet.String() }
	both := map[string]fileIdentity{key(conversion): held, key(directory): held}
	if location, present := presence(ctx, store, manifest, both, nil); !present || location != directory {
		t.Errorf("a component its directory and its conversion both hold is served from %q (present %t), want its directory", location, present)
	}
	fileOnly := map[string]fileIdentity{key(conversion): held, key(directory): {present: true, id: testutil.ArtifactBytesID(t, artifact.KindTensorSet, []byte("other")), size: 5}}
	if location, present := presence(ctx, store, manifest, fileOnly, nil); !present || location != conversion {
		t.Errorf("a component its directory does not hold is served from %q (present %t), want its file", location, present)
	}
}
