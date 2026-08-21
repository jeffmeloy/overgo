package artifact

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestTypedIdentityBoundaryContract(t *testing.T) {
	payload := []byte("same bytes")
	model, err := IdentifyBytes(KindModel, payload)
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := IdentifyBytes(KindDataset, payload)
	if err != nil {
		t.Fatal(err)
	}
	if model == dataset || model.Kind() != KindModel || dataset.Kind() != KindDataset {
		t.Fatalf("kind-qualified identities collapsed: model=%s dataset=%s", model, dataset)
	}

	weights, err := IdentifyBytes(KindTensorSet, []byte("weights"))
	if err != nil {
		t.Fatal(err)
	}
	components := []Component{{Role: ComponentWeights, Name: "weights", Artifact: weights}}
	manifest, err := NewManifest(KindModel, components)
	if err != nil {
		t.Fatal(err)
	}
	slices.Reverse(components)
	rebuilt, err := NewManifest(KindModel, components)
	if err != nil {
		t.Fatal(err)
	}
	if rebuilt.ID != manifest.ID {
		t.Fatalf("canonical manifest identities differ: %s != %s", rebuilt.ID, manifest.ID)
	}

	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	location, err := CanonicalLocalLocation(manifest.ID, LocationFile, path)
	if err != nil {
		t.Fatal(err)
	}
	if location.Artifact != manifest.ID || location.Value == manifest.ID.String() {
		t.Fatalf("physical location replaced content identity: %+v", location)
	}
	if err := manifest.Validate(); err != nil {
		t.Fatalf("location changed immutable manifest: %v", err)
	}
}
