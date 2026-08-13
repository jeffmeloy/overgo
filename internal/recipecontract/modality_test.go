package recipecontract_test

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestTypedImageRecipeSelectsRuntimeWithoutPlacement(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "typed-image-modality")
	tests := []struct {
		define func(artifact.ID) (recipe.Definition, error)
		want   []recipecontract.Modality
	}{
		{modelrecipe.LatentImageDefinition, []recipecontract.Modality{recipecontract.ModalityText}},
		{modelrecipe.OscillatorImageDefinition, []recipecontract.Modality{recipecontract.ModalityTable}},
		{modelrecipe.RoutedImageDefinition, []recipecontract.Modality{recipecontract.ModalityText}},
	}
	for _, test := range tests {
		definition, err := test.define(modelID)
		if err != nil {
			t.Fatal(err)
		}
		before, err := recipecontract.CompileModalitySignature(definition)
		if err != nil {
			t.Fatal(err)
		}
		for index := range definition.Nodes {
			definition.Nodes[index].Placement = recipe.PlacementDevice
		}
		after, err := recipecontract.CompileModalitySignature(definition)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(before.Inputs, test.want) || !slices.Equal(after.Inputs, test.want) || !slices.Equal(before.Outputs, after.Outputs) {
			t.Fatalf("modality before=%+v after=%+v want inputs=%v", before, after, test.want)
		}
	}
}
