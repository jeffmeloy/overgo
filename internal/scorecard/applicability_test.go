package scorecard

import (
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
)

// TestScorecardStatesInapplicableMetrics holds the scorecard to stating,
// not leaving blank, a metric a task's output cannot have: a capability
// that returns its output whole has no decode rate, and a training session
// no first output. A token-producing task's missing rate stays a gap a
// measurement must fill.
func TestScorecardStatesInapplicableMetrics(t *testing.T) {
	t.Parallel()
	var none evaluation.BenchmarkSummary
	for _, task := range []recipe.Task{recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskSpeech, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskImageEmbedding, recipe.TaskReconstruction, recipe.TaskTranscription} {
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
	// A training session decodes no output tokens, whatever else it measured.
	if cell := throughputCell(recipe.TaskTraining, none, false, &servingSummary{decodeTokens: 10, decodeNS: 1}); !cell.NotApplicable || cell.Reason == "" {
		t.Fatalf("training throughput = %+v, want a stated inapplicability", cell)
	}
	// A projection cell no attempt measured states why it is not a measurement missing; one an attempt measured stays.
	row := Row{Task: recipe.TaskProjection, Cells: map[metric.Name]Cell{
		metric.FirstOutput: gap("no serving prefill sample"), metric.Throughput: gap("no decode phase"), metric.Failures: gap("no attempts"),
		metric.Load: {Value: 500, Unit: "ms"}, metric.PeakMemory: gap("no peak"), metric.Quality: gap("no held-out evaluation"),
	}}
	stateNotApplicable(&row)
	for _, column := range []metric.Name{metric.FirstOutput, metric.Throughput, metric.Failures, metric.PeakMemory} {
		if cell := row.Cells[column]; !cell.NotApplicable || cell.Reason != projectionInRequest {
			t.Errorf("projection %s = %+v, want it stated as in the model's request", column, cell)
		}
	}
	if load := row.Cells[metric.Load]; !load.Measured() || load.NotApplicable {
		t.Errorf("a measured projection load = %+v, want it kept", load)
	}
	if quality := row.Cells[metric.Quality]; quality.NotApplicable {
		t.Errorf("projection held-out quality = %+v, want it left to its own evaluation", quality)
	}
	// A vqa row has no load phase of its own, and says so only where no attempt timed one; its other cells stay gaps.
	vqa := Row{Task: recipe.TaskVQA, Cells: map[metric.Name]Cell{metric.Load: gap("no recorded load phase"), metric.FirstOutput: gap("no prefill sample")}}
	stateNotApplicable(&vqa)
	if load := vqa.Cells[metric.Load]; !load.NotApplicable || load.Reason != vqaStreamsWeights {
		t.Errorf("vqa load = %+v, want it stated as part of its first output", load)
	}
	if first := vqa.Cells[metric.FirstOutput]; first.NotApplicable {
		t.Errorf("vqa first output = %+v, want a gap to measure", first)
	}
	timed := Row{Task: recipe.TaskVQA, Cells: map[metric.Name]Cell{metric.Load: {Value: 900, Unit: "ms"}}}
	stateNotApplicable(&timed)
	if load := timed.Cells[metric.Load]; load.NotApplicable || !load.Measured() {
		t.Errorf("a vqa load an attempt timed = %+v, want it kept", load)
	}
}
