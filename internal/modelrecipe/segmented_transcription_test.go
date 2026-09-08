package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestSegmentedTranscriptionTopology(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	base, err := TranscriptionDefinition(id(artifact.KindModel, "recognizer"), id(artifact.KindProfile, "format"), id(artifact.KindProfile, "recognition"), id(artifact.KindTokenizer, "tokens"), id(artifact.KindTensorInventory, "recognition-tensors"))
	if err != nil {
		t.Fatal(err)
	}
	activity, err := ActivityDefinition(id(artifact.KindModel, "activity"), id(artifact.KindProfile, "boundaries"), id(artifact.KindTensorInventory, "activity-tensors"))
	if err != nil {
		t.Fatal(err)
	}
	definition, err := SegmentedTranscriptionDefinition(base, activity)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	components, err := CompileComponentSessionPlanWithExtents(t.Context(), program, func(_ context.Context, model artifact.ID) (uint64, error) {
		if model != base.Model && model != activity.Model {
			t.Fatalf("foreign model %s", model)
		}
		return 1, nil
	})
	if err != nil || len(components.Components) != 2 || components.ArtifactBytes != 2 || components.Components[0].Model != activity.Model || components.Components[1].Model != base.Model {
		t.Fatalf("component plan %+v: %v", components, err)
	}
	gotBase, gotActivity, err := TranscriptionComponents(definition)
	if err != nil || gotBase.ID != base.ID || gotActivity.ID != activity.ID {
		t.Fatalf("component definitions: %v", err)
	}
	for slot, want := range []artifact.ID{base.ID, activity.ID} {
		if got, ok := definition.Dependency(recipe.DependencyExecutionRecipe, uint32(slot)); !ok || got != want {
			t.Fatal("component recipe dependency absent")
		}
	}
	for _, mutate := range []func(*recipe.Definition){
		func(d *recipe.Definition) { d.Edges = nil },
		func(d *recipe.Definition) { d.Nodes[0].ModelSlot = 0 },
		func(d *recipe.Definition) {
			d.Dependencies = append(d.Dependencies, recipe.Dependency{Role: recipe.DependencyProfile, Slot: 2, Artifact: id(artifact.KindProfile, "extra")})
		},
		func(d *recipe.Definition) {
			for i := range d.Dependencies {
				if d.Dependencies[i].Role == recipe.DependencyExecutionRecipe {
					d.Dependencies[i].Artifact = activity.ID
				}
			}
		},
	} {
		// Reconstruct before mutation so each case owns its slices.
		changed, err := SegmentedTranscriptionDefinition(base, activity)
		if err != nil {
			t.Fatal(err)
		}
		mutate(&changed)
		changed, err = recipe.NewDefinitionWithDependencies(changed.Task, changed.Dependencies, changed.Nodes, changed.Edges, changed.Inputs, changed.Outputs)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := TranscriptionComponents(changed); err == nil {
			t.Fatal("altered topology admitted")
		}
	}
}
