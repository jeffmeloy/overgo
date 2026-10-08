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
// carries when a use is recorded for its model and task and it was scored on
// the measure the use names: the cell says what it measures in place of the
// use, and its text says it is not a measure of what the model is for. A set
// built for that measure, whose name carries its cases after the source, is
// marked as the measure is. The same model under another task, a model with
// no recorded use, and a cell that reads nothing carry no mark, and their text
// is what it was.
//
// A reading of the model on another measure is not marked, and is the one read
// first: where the headline is marked and another of the row's readings is
// not, the two change places, and both are still the row's.
func TestReadingOutsideItsModelsUseSaysSo(t *testing.T) {
	t.Parallel()
	reading := func(name string, value float64, measure string) Cell {
		return Cell{Value: value, Label: "accuracy, " + measure, Source: testutil.ArtifactID(t, artifact.KindEvaluation, name+measure), Cases: 128, Measure: measure}
	}
	row := func(name string, task recipe.Task, headline Cell, others ...Cell) Row {
		return Row{Name: name, Task: task, Cells: map[metric.Name]Cell{metric.Quality: headline}, Heldout: others}
	}
	rows := []Row{
		row("a drafter", recipe.TaskInference, reading("a drafter", 910, "store/dna")),
		row("a drafter", recipe.TaskTraining, reading("a drafter trained", 0.5, "store/dna")),
		row("an assistant", recipe.TaskInference, reading("an assistant", 0.5, "store/dna")),
		row("a model nothing documents", recipe.TaskImageGen, inapplicable("nothing measures it")),
		row("an editor", recipe.TaskVideoGen, reading("an editor", 0.25, "clips/whole-change 0a1b"),
			reading("an editor, an earlier build", 0.2, "clips/whole-change 0a1b"), reading("an editor", 0.75, "clips/one-thing 2c3d")),
		row("a drawer", recipe.TaskImageGen, reading("a drawer", 0.5, "captions/photographs 4e5f")),
	}
	markUses(rows, []ModelUse{
		{Model: "a drafter", Task: recipe.TaskInference, Measure: "store/dna", Use: "to draft", Source: "its card", Measured: "its perplexity alone", Needs: "the larger model"},
		{Model: "a model nothing documents", Task: recipe.TaskImageGen, Use: "not stated", Source: "no documentation", Measured: "nothing", Needs: "the owner to say"},
		{Model: "an editor", Task: recipe.TaskVideoGen, Measure: "clips/whole-change", Use: "to change one thing", Source: "its card", Measured: "changes of the whole clip", Needs: "nothing"},
		{Model: "a drawer", Task: recipe.TaskImageGen, Measure: "captions/photo", Use: "to draw", Source: "its card", Measured: "something", Needs: "nothing"},
	})
	marked := rows[0].Cells[metric.Quality]
	if marked.OutsideUse != "its perplexity alone" || marked.QualityText() != "910 on 128 cases, not a measure of what the model is for (accuracy, store/dna)" {
		t.Fatalf("the drafter's reading carries %q and reads %q", marked.OutsideUse, marked.QualityText())
	}
	for _, at := range []int{1, 2, 3, 5} {
		cell := rows[at].Cells[metric.Quality]
		if cell.OutsideUse != "" || cell.Measured() && strings.Contains(cell.QualityText(), "what the model is for") {
			t.Errorf("%s %s is marked outside its use: %+v", rows[at].Name, rows[at].Task, cell)
		}
	}
	if rows[3].Cells[metric.Quality].Reason != "nothing measures it" {
		t.Fatalf("a cell that reads nothing lost its reason: %+v", rows[3].Cells[metric.Quality])
	}
	editor := rows[4]
	if headline := editor.Cells[metric.Quality]; headline.Value != 0.75 || headline.OutsideUse != "" {
		t.Fatalf("the editor's headline is %+v, want its reading on the measure of its use", headline)
	}
	if len(editor.Heldout) != 2 || editor.Heldout[0].OutsideUse == "" || editor.Heldout[1].OutsideUse == "" ||
		editor.Heldout[0].Value+editor.Heldout[1].Value != 0.25+0.2 {
		t.Fatalf("the editor's other readings are %+v, want both builds of the whole-clip measure, marked", editor.Heldout)
	}
}
