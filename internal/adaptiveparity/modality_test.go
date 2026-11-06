package adaptiveparity

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestTypedImageRecipeSelectsRuntimeWithoutPlacement(t *testing.T) {
	t.Parallel()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "typed-image-modality")
	profileID := testutil.ArtifactID(t, artifact.KindProfile, "typed-image-profile")
	tests := []struct {
		prepare recipe.ModuleID
		profile artifact.ID
		want    []recipecontract.Modality
	}{
		{modelrecipe.ModuleLatentImagePrepare, profileID, []recipecontract.Modality{recipecontract.ModalityText}},
		{modelrecipe.ModuleOscillatorImagePrepare, artifact.ID{}, []recipecontract.Modality{recipecontract.ModalityTable}},
		{modelrecipe.ModuleRoutedImagePrepare, profileID, []recipecontract.Modality{recipecontract.ModalityText}},
	}
	for _, test := range tests {
		definition := testcheck.Must(modelrecipe.GenerationDefinition(test.prepare, modelID, test.profile))(t)
		before := testcheck.Must(compileModalitySignature(definition))(t)
		for index := range definition.Nodes {
			definition.Nodes[index].Placement = recipe.PlacementDevice
		}
		after := testcheck.Must(compileModalitySignature(definition))(t)
		if !slices.Equal(before.Inputs, test.want) || !slices.Equal(after.Inputs, test.want) || !slices.Equal(before.Outputs, after.Outputs) {
			t.Fatalf("modality before=%+v after=%+v want inputs=%v", before, after, test.want)
		}
	}
}

func TestTypedVideoRecipeSelectsSemanticsWithoutPlacement(t *testing.T) {
	t.Parallel()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "typed-video-modality")
	profileID := testutil.ArtifactID(t, artifact.KindProfile, "typed-video-profile")
	tests := []struct {
		define func(artifact.ID, artifact.ID) (recipe.Definition, error)
		want   []recipecontract.Modality
	}{
		{func(model, profile artifact.ID) (recipe.Definition, error) {
			return modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentVideoPrepare, model, profile)
		}, []recipecontract.Modality{recipecontract.ModalityText}},
		{modelrecipe.ReferenceVideoEditDefinition, []recipecontract.Modality{recipecontract.ModalityText, recipecontract.ModalityVideo}},
	}
	for _, test := range tests {
		definition := testcheck.Must(test.define(modelID, profileID))(t)
		before := testcheck.Must(compileModalitySignature(definition))(t)
		for index := range definition.Nodes {
			definition.Nodes[index].Placement = recipe.PlacementDevice
		}
		after := testcheck.Must(compileModalitySignature(definition))(t)
		if !slices.Equal(before.Inputs, test.want) || !slices.Equal(after.Inputs, test.want) || !slices.Equal(before.Outputs, []recipecontract.Modality{recipecontract.ModalityVideo}) || !slices.Equal(before.Outputs, after.Outputs) {
			t.Fatalf("modality before=%+v after=%+v want inputs=%v", before, after, test.want)
		}
	}
}
