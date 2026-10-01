package scorecard

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestQualityReadsEachTasksHeldoutEvidence holds held-out quality to the
// evidence each task has. A projection recipe the evaluation circuit scored
// on sets outside the derived catalog reads one of them under a full
// contract: its metric and direction, the evaluation origin, the set's
// identity as suite and the protocol it ran under, and the same set
// whichever ran last. A recipe's wall-time record carries no evaluation
// evidence and is not held-out quality. A task whose held-out evaluation is
// only proposed reads not applicable with the proposal, is proposed as work,
// and withholds no promotion; an inference recipe nothing evaluated is a
// gap that withholds it.
func TestQualityReadsEachTasksHeldoutEvidence(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	projection, timed := id(artifact.KindRecipe, "projection"), id(artifact.KindRecipe, "timed")
	first, second, timing := id(artifact.KindDataset, "caption-matching-a"), id(artifact.KindDataset, "caption-matching-b"), id(artifact.KindDataset, "smoke-timing")
	if second.String() < first.String() {
		first, second = second, first
	}
	// One run per set: a run has one metric record.
	run := func(set artifact.ID) artifact.ID { return id(artifact.KindRun, "run "+set.String()) }
	if _, err := store.Commit(ctx, artifact.Batch{Key: "quality/authorities", Artifacts: []artifact.Descriptor{
		{ID: projection}, {ID: timed}, {ID: first}, {ID: second}, {ID: timing}, {ID: run(first)}, {ID: run(second)}, {ID: run(timing)},
	}}); err != nil {
		t.Fatal(err)
	}
	accuracy := func(value float64) []runrecord.Metric {
		return []runrecord.Metric{{Name: "accuracy", Value: value, Direction: runrecord.DirectionMaximize}}
	}
	record := func(recipeID, set artifact.ID, metrics []runrecord.Metric) runrecord.Evaluation {
		t.Helper()
		evaluated, err := runrecord.NewEvaluation(recipeID, run(set), set, metrics)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := evaluated.Batch("quality/" + set.String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return evaluated
	}
	// scored records an evaluation the circuit ran, with the evidence that
	// binds it to a plan over the set.
	scored := func(recipeID, set artifact.ID, value float64) runrecord.Evaluation {
		t.Helper()
		evaluated := record(recipeID, set, accuracy(value))
		data, err := json.Marshal(evaluation.EvaluationEvidence{
			Version: artifact.InitialDocumentVersion,
			Plan:    id(artifact.KindProfile, "plan "+set.String()), Acceptance: id(artifact.KindProfile, "acceptance"),
			Evaluator: id(artifact.KindEvidence, "evaluator"), Report: id(artifact.KindEvaluation, "report "+set.String()),
			Run: run(set), Evaluation: evaluated.ID, ModelDefinition: id(artifact.KindModelDefinition, "definition"),
			Recipe: recipeID, Dataset: set, Split: id(artifact.KindDatasetShard, "split "+set.String()),
			Environment: id(artifact.KindEvidence, "environment"), CodeCommit: strings.Repeat("ab", 20),
			Phases:  []runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: 1}},
			Metrics: accuracy(value),
		})
		if err != nil {
			t.Fatal(err)
		}
		evidence, err := evaluation.ParseEvaluationEvidence(data)
		if err != nil {
			t.Fatal(err)
		}
		content, err := evidence.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch, err := artifact.NewDocumentBatch("quality/evidence/"+set.String(), []artifact.Content{content}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return evaluated
	}
	// The set named second runs last, so it is the newest.
	read := scored(projection, first, 0.5)
	scored(projection, second, 0.75)
	record(timed, timing, []runrecord.Metric{{Name: "smoke_wall_ns", Value: 1, Direction: runrecord.DirectionMinimize}})

	suites := map[artifact.ID]string{}
	if err := nameEvaluatedSets(ctx, store, suites); err != nil {
		t.Fatal(err)
	}
	index := evaluation.LatestEvidence(ctx, store, suites, 0)
	cell := qualityCell(recipe.TaskProjection, index.EvaluationsByRecipe[projection], nil)
	want := Contract{Metric: "accuracy", Origin: originEvaluation, Suite: first.String(), Protocol: evaluation.PromptingRawCompletion.Label()}
	if !cell.Measured() || cell.Value != 0.5 || cell.Source != read.ID || cell.Direction != runrecord.DirectionMaximize || cell.Contract != want {
		t.Fatalf("projection quality = %+v, want the set named first, %+v, from its evaluation", cell, want)
	}
	if cell := qualityCell(recipe.TaskVQA, index.EvaluationsByRecipe[timed], nil); cell.Measured() || !cell.NotApplicable || !strings.Contains(cell.Reason, "proposed: ") {
		t.Fatalf("a wall-time record reads as held-out quality %+v, want the task's proposed evaluation", cell)
	}

	row := func(name string, task recipe.Task, quality Cell) Row {
		cells := map[metric.Name]Cell{}
		for _, column := range metric.Columns {
			cells[column] = inapplicable("not measured in this fixture")
		}
		cells[metric.Quality] = quality
		return Row{Model: id(artifact.KindModel, name), Name: name, Task: task, Cells: cells}
	}
	proposed := qualityCell(recipe.TaskImageGen, nil, nil)
	published := Record{Commit: "b", Rows: []Row{row("projector", recipe.TaskProjection, cell), row("painter", recipe.TaskImageGen, proposed)}}
	baseline := Record{Commit: "a"}
	if withheld := published.Withheld(&baseline); withheld != "" {
		t.Fatalf("a task whose held-out evaluation is only proposed withholds promotion: %s", withheld)
	}
	gaps := published.Gaps(&baseline)
	if len(gaps) != 1 || gaps[0].Key != "scorecard-unscored-image-gen" || !strings.Contains(gaps[0].Detail, "painter held-out-quality: "+proposed.Reason) {
		t.Fatalf("gaps = %+v, want one row to define image-gen's held-out evaluation", gaps)
	}
	unevaluated := qualityCell(recipe.TaskInference, nil, nil)
	published.Rows = append(published.Rows, row("reader", recipe.TaskInference, unevaluated))
	if unevaluated.NotApplicable || unevaluated.Measured() || !strings.Contains(published.Withheld(&baseline), "reader has no inference held-out quality") {
		t.Fatalf("an inference recipe nothing evaluated reads %+v and withholds %q, want a gap that withholds", unevaluated, published.Withheld(&baseline))
	}
}
