package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestReadingOutsideItsModelsUseSaysSo holds the mark a held-out quality
// carries when a use is recorded for its model and task: the cell says what
// it measures in place of the use, and its text says it is not a measure of
// what the model is for. The same model under another task, a model with no
// recorded use, and a cell that reads nothing carry no mark, and their text
// is what it was.
func TestReadingOutsideItsModelsUseSaysSo(t *testing.T) {
	t.Parallel()
	measured := func(name string, task recipe.Task) Row {
		return Row{Name: name, Task: task, Cells: map[metric.Name]Cell{metric.Quality: {
			Value: 910, Label: "perplexity, a corpus", Source: testutil.ArtifactID(t, artifact.KindEvaluation, name+string(task)), Cases: 128,
		}}}
	}
	rows := []Row{
		measured("a drafter", recipe.TaskInference),
		measured("a drafter", recipe.TaskTraining),
		measured("an assistant", recipe.TaskInference),
		{Name: "a model nothing documents", Task: recipe.TaskImageGen, Cells: map[metric.Name]Cell{metric.Quality: inapplicable("nothing measures it")}},
	}
	markUses(rows, []ModelUse{
		{Model: "a drafter", Task: recipe.TaskInference, Use: "to draft", Source: "its card", Measured: "its perplexity alone", Needs: "the larger model"},
		{Model: "a model nothing documents", Task: recipe.TaskImageGen, Use: "not stated", Source: "no documentation", Measured: "nothing", Needs: "the owner to say"},
	})
	marked := rows[0].Cells[metric.Quality]
	if marked.OutsideUse != "its perplexity alone" || marked.QualityText() != "910 on 128 cases, not a measure of what the model is for (perplexity, a corpus)" {
		t.Fatalf("the drafter's reading carries %q and reads %q", marked.OutsideUse, marked.QualityText())
	}
	for _, row := range rows[1:] {
		cell := row.Cells[metric.Quality]
		if cell.OutsideUse != "" || cell.Measured() && strings.Contains(cell.QualityText(), "what the model is for") {
			t.Errorf("%s %s is marked outside its use: %+v", row.Name, row.Task, cell)
		}
	}
	if rows[3].Cells[metric.Quality].Reason != "nothing measures it" {
		t.Fatalf("a cell that reads nothing lost its reason: %+v", rows[3].Cells[metric.Quality])
	}
}
