package adaptiveparity

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

func TestInferenceModalityMatrixDerivedFromRecipes(t *testing.T) {
	model := identify(t, artifact.KindModel, "modality-model")
	definitions := make([]recipe.Definition, 0, 7)
	for _, task := range []recipe.Task{
		recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
		recipe.TaskSpeech, recipe.TaskImageGen, recipe.TaskVQA,
	} {
		definition, err := modelrecipe.CapabilityDefinition(task, model)
		if err != nil {
			t.Fatal(err)
		}
		definitions = append(definitions, definition)
	}
	latentImage, err := modelrecipe.CapabilityDefinitionAt(recipe.TaskImageGen, model, recipe.PlacementHybrid)
	if err != nil {
		t.Fatal(err)
	}
	definitions = append(definitions, latentImage)

	required := []Contract{
		{Task: recipe.TaskForecast, Signature: Signature{Inputs: []Modality{ModalityTimeSeries}, Outputs: []Modality{ModalityTimeSeries}}},
		{Task: recipe.TaskTabular, Signature: Signature{Inputs: []Modality{ModalityTable}, Outputs: []Modality{ModalityTable}}},
		{Task: recipe.TaskSeq2Seq, Signature: Signature{Inputs: []Modality{ModalityText}, Outputs: []Modality{ModalityText}}},
		{Task: recipe.TaskSpeech, Signature: Signature{Inputs: []Modality{ModalityText}, Outputs: []Modality{ModalityAudio}}},
		{Task: recipe.TaskImageGen, Signature: Signature{Inputs: []Modality{ModalityTable}, Outputs: []Modality{ModalityImage}}},
		{Task: recipe.TaskImageGen, Signature: Signature{Inputs: []Modality{ModalityText}, Outputs: []Modality{ModalityImage}}},
		{Task: recipe.TaskVQA, Signature: Signature{Inputs: []Modality{ModalityImage, ModalityText}, Outputs: []Modality{ModalityText}}},
		{Task: recipe.TaskVideoGen, Signature: Signature{Inputs: []Modality{ModalityText}, Outputs: []Modality{ModalityVideo}}},
	}
	rows, err := InferenceModalityMatrix(definitions, required)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(required) {
		t.Fatalf("matrix rows = %d, want %d", len(rows), len(required))
	}
	refused := 0
	for _, row := range rows {
		if row.Status == ContractRefused {
			refused++
			if row.Task != recipe.TaskVideoGen || row.Recipe.Valid() || row.Reason == "" {
				t.Fatalf("invalid refusal row: %+v", row)
			}
			continue
		}
		if row.Status != ContractSupported || !row.Recipe.Valid() {
			t.Fatalf("invalid supported row: %+v", row)
		}
	}
	if refused != 1 {
		t.Fatalf("refused rows = %d, want 1", refused)
	}
}
