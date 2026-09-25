package discovery

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestMovedModelResolvesByIdentity moves a registered model's weights from
// their recorded directory into the data root's models directory: discovery
// finds them there by their recorded identity, and refuses different bytes
// at the same relative path.
// Serial: sets OVERGO_DATA_ROOT.
func TestMovedModelResolvesByIdentity(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	payload := []byte("moved-model-weights")
	component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
	recorded := filepath.Join(t.TempDir(), "models", "family-1.0", "head", "model.safetensors")
	writeFile(t, recorded, payload)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: component,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/moved/facts",
		Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(payload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{
			Artifact: component, Kind: artifact.LocationFile, Value: recorded,
		}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	base := t.TempDir()
	t.Setenv(dataroot.Env, base)
	if err := os.Remove(recorded); err != nil {
		t.Fatal(err)
	}
	moved := filepath.Join(base, "models", "family-1.0", "head", "model.safetensors")
	writeFile(t, moved, []byte("different-model-bytes"))
	if location, present := presence(ctx, store, manifest, map[string]fileIdentity{}, nil); present {
		t.Fatalf("different bytes at %s resolved as the model", location)
	}
	writeFile(t, moved, payload)
	location, present := presence(ctx, store, manifest, map[string]fileIdentity{}, nil)
	if !present || location != moved {
		t.Fatalf("presence = %q, %t; want %q, true", location, present, moved)
	}
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
