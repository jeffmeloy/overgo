package artifact

import (
	"slices"
	"testing"
)

func TestManifestIdentityUsesCanonicalComponents(t *testing.T) {
	weights, _ := IdentifyBytes(KindTensorSet, []byte("weights"))
	config, _ := IdentifyBytes(KindFile, []byte("config"))
	components := []Component{
		{Role: ComponentWeights, Name: "weights", Artifact: weights},
		{Role: ComponentConfig, Name: "config", Artifact: config},
	}
	first, err := NewManifest(KindModel, components)
	if err != nil {
		t.Fatal(err)
	}
	slices.Reverse(components)
	second, err := NewManifest(KindModel, components)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID || !slices.Equal(first.Components, second.Components) {
		t.Fatalf("canonical manifests differ: %s != %s", first.ID, second.ID)
	}
	descriptor, err := first.Descriptor()
	if err != nil {
		t.Fatal(err)
	}
	if descriptor.ID != first.ID || descriptor.MediaType != ManifestMediaType || descriptor.Schema != ManifestSchema {
		t.Fatalf("descriptor = %+v", descriptor)
	}
	cloned := first.Clone()
	cloned.Components[0].Name = "changed"
	if first.Components[0].Name == cloned.Components[0].Name {
		t.Fatal("manifest clone retained component storage")
	}
}

func TestManifestRejectsDuplicateSlotsAndNames(t *testing.T) {
	first, _ := IdentifyBytes(KindFile, []byte("first"))
	second, _ := IdentifyBytes(KindFile, []byte("second"))
	for _, components := range [][]Component{
		{
			{Role: ComponentConfig, Name: "first", Artifact: first},
			{Role: ComponentConfig, Name: "second", Artifact: second},
		},
		{
			{Role: ComponentConfig, Name: "same", Artifact: first},
			{Role: ComponentCompanion, Name: "same", Artifact: second},
		},
	} {
		if _, err := NewManifest(KindModel, components); err == nil {
			t.Fatalf("accepted components %+v", components)
		}
	}
}

func TestManifestIdentityChangesWithComponentContent(t *testing.T) {
	first, _ := IdentifyBytes(KindTensorSet, []byte("first"))
	second, _ := IdentifyBytes(KindTensorSet, []byte("second"))
	makeManifest := func(id ID) Manifest {
		manifest, err := NewManifest(KindModel, []Component{{
			Role: ComponentWeights, Name: "weights", Artifact: id,
		}})
		if err != nil {
			t.Fatal(err)
		}
		return manifest
	}
	if makeManifest(first).ID == makeManifest(second).ID {
		t.Fatal("component content did not change manifest identity")
	}
}
