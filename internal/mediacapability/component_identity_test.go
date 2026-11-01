package mediacapability

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/testutil"
)

func TestImageComponentIdentityExcludesSiblingLayout(t *testing.T) {
	var members []artifact.Component
	for ordinal, directory := range []string{"text_encoder", "transformer", "vae"} {
		for _, role := range []artifact.ComponentRole{artifact.ComponentConfig, artifact.ComponentWeights} {
			name := directory + "/" + role.String()
			kind := artifact.KindFile
			if role == artifact.ComponentWeights {
				kind = artifact.KindTensorSet
			}
			members = append(members, artifact.Component{Role: role, Ordinal: uint32(ordinal), Name: name, Artifact: testutil.ArtifactID(t, kind, name)})
		}
	}
	source := modelrecipe.GenerationSourceProfile{Prepare: modelrecipe.ModuleLatentImagePrepare}
	decoder := func(components []artifact.Component) artifact.ID {
		t.Helper()
		parent, err := artifact.NewManifest(artifact.KindModel, components)
		if err != nil {
			t.Fatal(err)
		}
		before := slices.Clone(parent.Components)
		bindings, _, err := latentImageComponentBindings(source, modelartifact.Inventory{Manifest: parent})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(parent.Components, before) {
			t.Fatal("binding changed the parent manifest")
		}
		for _, binding := range bindings {
			if binding.Node == "decode" {
				return binding.Model
			}
		}
		t.Fatal("missing decoder binding")
		return artifact.ID{}
	}
	first := decoder(members)
	moved := slices.Clone(members)
	for i := range moved {
		if moved[i].Name == "vae/config" {
			moved[i].Ordinal++
		}
	}
	moved = append(moved, artifact.Component{Role: artifact.ComponentConfig, Ordinal: 2, Name: "transformer/extra-policy", Artifact: testutil.ArtifactID(t, artifact.KindFile, "other component policy")})
	if second := decoder(moved); second != first {
		t.Fatalf("unchanged decoder inherited sibling layout: %s != %s", first, second)
	}
	slices.Reverse(moved)
	if decoder(moved) != first {
		t.Fatal("component enumeration changed decoder identity")
	}
	for i := range moved {
		if moved[i].Name == "vae/config" {
			moved[i].Artifact = testutil.ArtifactID(t, artifact.KindFile, "changed decoder policy")
		}
	}
	if decoder(moved) == first {
		t.Fatal("decoder policy change retained the component identity")
	}
}
