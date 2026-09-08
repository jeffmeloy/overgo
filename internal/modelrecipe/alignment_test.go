package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestAlignmentTopology(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	base, err := TranscriptionDefinition(id(artifact.KindModel, "recognizer"), id(artifact.KindProfile, "format"), id(artifact.KindProfile, "recognition"), id(artifact.KindTokenizer, "tokens"), id(artifact.KindTensorInventory, "tensors"))
	if err != nil {
		t.Fatal(err)
	}
	profile := id(artifact.KindProfile, "alignment")
	definition, err := AlignmentDefinition(base, profile)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileCapability(definition); err != nil {
		t.Fatal(err)
	}
	got, activity, err := SpeechComponents(definition)
	if err != nil || got.ID != base.ID || activity.ID.Valid() {
		t.Fatalf("components: %+v %+v %v", got, activity, err)
	}
	adapted, err := AdaptedTranscriptionDefinition(base, id(artifact.KindCheckpoint, "adapter"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := AlignmentDefinition(adapted, profile); err == nil {
		t.Fatal("adapted alignment admitted")
	}
	if _, err := AlignmentDefinition(definition, profile); err == nil {
		t.Fatal("nested alignment admitted")
	}
	if _, err := AlignmentDefinition(base, base.ID); err == nil {
		t.Fatal("non-profile admitted")
	}
	for _, mutate := range []func(*recipe.Definition){
		func(d *recipe.Definition) { d.Nodes[0].Residency = recipe.ResidencyStream },
		func(d *recipe.Definition) {
			d.Dependencies = append(d.Dependencies, recipe.Dependency{Role: recipe.DependencyProfile, Slot: 1, Artifact: profile})
		},
		func(d *recipe.Definition) {
			for i := range d.Dependencies {
				if d.Dependencies[i].Role == recipe.DependencyExecutionRecipe {
					d.Dependencies[i].Artifact = definition.ID
				}
			}
		},
	} {
		changed, err := AlignmentDefinition(base, profile)
		if err != nil {
			t.Fatal(err)
		}
		mutate(&changed)
		changed, err = recipe.NewDefinitionWithDependencies(changed.Task, changed.Dependencies, changed.Nodes, changed.Edges, changed.Inputs, changed.Outputs)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := SpeechComponents(changed); err == nil {
			t.Fatal("altered complete topology admitted")
		}
	}
}
