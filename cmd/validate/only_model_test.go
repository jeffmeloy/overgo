package main

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestOneModelsCellsAreMeasuredAlone holds validate -run -model to the named model's cells: its serving cell and its
// benchmark stay, in the buckets the plan put them in, another model's cells and every lane go, and a name no cell
// answers to is refused rather than measuring nothing.
func TestOneModelsCellsAreMeasuredAlone(t *testing.T) {
	t.Parallel()
	cell := func(name, validation string) ModelValidation {
		return ModelValidation{
			Model: testutil.ArtifactID(t, artifact.KindModel, name), ModelName: name + ".gguf", Location: "C:/models/" + name + ".gguf",
			Kind: recipe.TaskInference, Modality: "text", Validation: validation,
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, name),
		}
	}
	plan := SelectValidation(Inputs{Registered: []ModelValidation{
		cell("Qwen3.5-9B", servingValidation), cell("Qwen3.5-9B", "mmlu-pro"),
		cell("Bonsai-27B", servingValidation), cell("Bonsai-27B", "mmlu-pro"),
	}})
	plan.Suites = cycleSuites()
	declined := testutil.ArtifactID(t, artifact.KindModel, "Qwen3.5-9B")
	named := func(cell ModelValidation) bool { return cell.Model == declined }
	only, err := plan.OnlyModel("Qwen3.5-9B", named)
	if err != nil {
		t.Fatal(err)
	}
	if len(only.Suites) != 0 {
		t.Errorf("one model's measurement runs %d lanes, want none", len(only.Suites))
	}
	measured := map[string]int{}
	for _, bucket := range [][]Selected{only.Run, only.Background, only.OperatorOwned} {
		for _, sel := range bucket {
			measured[sel.ModelName]++
		}
	}
	if len(measured) != 1 || measured["Qwen3.5-9B.gguf"] != 2 {
		t.Errorf("one model's measurement holds %v, want Qwen3.5-9B.gguf's two cells alone", measured)
	}
	if len(only.Run)+len(only.Background) != 2 || len(plan.Run)+len(plan.Background) != 4 {
		t.Errorf("the plan's cells moved: %d of %d kept", len(only.Run)+len(only.Background), len(plan.Run)+len(plan.Background))
	}
	if _, err := plan.OnlyModel("Carbon-500M", func(ModelValidation) bool { return false }); err == nil || !strings.Contains(err.Error(), "Carbon-500M") {
		t.Errorf("a name no cell answers to gave %v, want a refusal naming it", err)
	}
}
