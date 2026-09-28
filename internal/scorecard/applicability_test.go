package scorecard

import (
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/recipe"
)

// TestScorecardStatesInapplicableMetrics holds the scorecard to stating,
// not leaving blank, a metric a task's output cannot have: a capability
// that returns its output whole has no decode rate, and a training session
// no first output. A token-producing task's missing rate stays a gap a
// measurement must fill.
func TestScorecardStatesInapplicableMetrics(t *testing.T) {
	t.Parallel()
	var none evaluation.BenchmarkSummary
	for _, task := range []recipe.Task{recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskSpeech, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskImageEmbedding} {
		if cell := throughputCell(task, none, false, nil); cell.Measured() || !cell.NotApplicable || cell.Reason == "" {
			t.Fatalf("%s throughput = %+v, want a stated inapplicability", task, cell)
		}
	}
	if cell := throughputCell(recipe.TaskInference, none, false, nil); cell.NotApplicable || cell.Measured() {
		t.Fatalf("inference throughput = %+v, want a gap to measure", cell)
	}
	if cell := firstOutputCell(recipe.TaskTraining, none, false, nil); !cell.NotApplicable {
		t.Fatalf("training first output = %+v, want a stated inapplicability", cell)
	}
}
