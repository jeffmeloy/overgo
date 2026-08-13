package adaptiveparity

import (
	"slices"
	"testing"

	"overgo/internal/recipe"
)

func TestInferenceModalityMatrixDerivedFromRecipes(t *testing.T) {
	rows, err := InferenceModalityMatrix()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(inferenceTasks) {
		t.Fatalf("rows = %d, want %d declared inference tasks", len(rows), len(inferenceTasks))
	}
	for _, test := range []struct {
		task      recipe.Task
		state     string
		signature string
	}{
		{recipe.TaskInference, "supported", "text->text"},
		{recipe.TaskForecast, "supported", "time-series->time-series"},
		{recipe.TaskTabular, "supported", "table->table"},
		{recipe.TaskSeq2Seq, "supported", "text->text"},
		{recipe.TaskSpeech, "supported", "text->audio"},
		{recipe.TaskImageGen, "supported", "table->image"},
		{recipe.TaskVQA, "supported", "image+text->text"},
		{recipe.TaskGeneration, "refused", "text->text"},
		{recipe.TaskEmbedding, "refused", "text->table"},
		{recipe.TaskRerank, "refused", "text+text->table"},
		{recipe.TaskProjection, "refused", "image->text"},
		{recipe.TaskVideoGen, "refused", "text->video"},
	} {
		row, ok := FindModalityRow(rows, test.task)
		if !ok {
			t.Fatalf("task %q absent", test.task)
		}
		if row.State != test.state || row.Signature() != test.signature {
			t.Errorf("task %q = %s %s, want %s %s", test.task, row.State, row.Signature(), test.state, test.signature)
		}
		if row.State == "supported" && row.RecipeID == "" {
			t.Errorf("task %q supported without recipe identity", test.task)
		}
		if row.State == "refused" && row.Reason == "" {
			t.Errorf("task %q refused without reason", test.task)
		}
	}
	if got := rows[len(rows)-1].Task; got != recipe.TaskVQA || !slices.Equal(rows[len(rows)-1].Inputs, []string{"image", "text"}) {
		t.Fatalf("ordered VQA boundary lost: %+v", rows[len(rows)-1])
	}
}
