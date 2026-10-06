package scorecard

import (
	"math"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestAServingCellIsJudgedByItsOwnSpread holds a serving timing to what its own repeats do: a recipe
// with four attempts that timed their load reads the newest and the spread of the newest three (the
// oldest reads in none), and a comparison allows a change up to that spread beside the
// column's allowance: a 9 ms operation that read 9, 16, 23, 49 and 18 ms on repeats is not a decline
// at 30 ms, a stable cell that repeats within two percent is one at 160 ms from 100, and a decline is
// restored when the cell is back within its own spread.
func TestAServingCellIsJudgedByItsOwnSpread(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "noisy-model")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "noisy-environment")
	noisy := testutil.ArtifactID(t, artifact.KindRecipe, "noisy-recipe")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "noisy/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: environment}, {ID: noisy}}}); err != nil {
		t.Fatal(err)
	}
	const millisecond = uint64(time.Millisecond)
	// Committed oldest first. Four attempts loaded their model; the oldest read 9 ms.
	for index, prefill := range []uint64{9, 16, 23, 49} {
		started := int64(index + 1)
		observation, err := runrecord.NewServingObservation(runrecord.ServingObservation{
			Model: model, Recipe: noisy, Environment: environment, Task: recipe.TaskInference,
			Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: started, MeasuredNS: prefill * millisecond,
			Hardware: []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: prefill * millisecond}},
			Phases:   []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: millisecond}},
		})
		if err != nil {
			t.Fatal(err)
		}
		batch, err := observation.Batch("noisy/" + observation.ID.String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	summaries, err := servingByRecipe(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	cell := firstOutputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, summaries[noisy])
	if want := 49.0; cell.Value != want || math.Abs(cell.Spread-(49.0-16)*3/(49+23+16)) > 1e-9 {
		t.Fatalf("first output = %v with spread %v, want the newest attempt %v and the range of 16, 23 and 49 ms over their mean", cell.Value, cell.Spread, want)
	}

	contract := Contract{Metric: "time to prefill", Origin: originServing}
	timing := func(value, spread float64) Cell {
		return Cell{Value: value, Spread: spread, Unit: "ms", Contract: contract}
	}
	for _, test := range []struct {
		name      string
		then, now Cell
		want      Verdict
	}{
		{"a 9 ms operation at 30 ms, its repeats spread by 1.4", timing(9, 0), timing(30, 1.43), holding},
		{"the same cell read with the old spread", timing(9, 1.43), timing(30, 0), holding},
		{"a stable cell at 160 ms from 100", timing(100, 0.02), timing(160, 0.02), regressed},
		{"a stable cell at 120 ms from 100", timing(100, 0.02), timing(120, 0.02), holding},
	} {
		if verdict, why := holds(metric.FirstOutput, test.then, test.now); verdict != test.want {
			t.Errorf("%s: holds = %q (%s), want %q", test.name, verdict, why, test.want)
		}
	}
	row := Row{Model: model, Task: recipe.TaskInference, Cells: map[metric.Name]Cell{metric.FirstOutput: timing(30, 1.43)}}
	key := regressionKey(row, metric.FirstOutput, row.Cells[metric.FirstOutput])
	if restored, found := Restored([]Row{row}, key, 9); !restored || !found {
		t.Errorf("a cell within its own spread of the target = (restored %t, found %t), want restored", restored, found)
	}
}
