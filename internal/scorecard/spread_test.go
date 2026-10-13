package scorecard

import (
	"math"
	"slices"
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

// TestACellsOwnSpreadNeitherHidesItsDeclineNorRestoresIt holds a serving timing's decline to what was
// measured before the candidate existed. A recipe with four attempts that timed their load reads the
// median of the newest three and how far they stand from it (the oldest reads in neither): attempts of
// 16, 23 and 49 ms publish 23 ms with a spread of 7/23, and attempts of 100, 100 and 160 ms publish
// 100 ms with none, the one slow attempt moving neither. That spread is stated beside the reading and
// judges nothing: what a later reading is allowed is its reference's, taken from the cell's earlier
// readings, and never the candidate's own spread nor the reference's published one. The rows that hold
// that set their spreads and allowances by hand: a cell at 160 ms that carries a spread of a half is a
// decline from a stable 100, a 9 ms operation at 30 ms is a decline whatever spread either cell
// publishes and holds only once its reference's readings allow that much, and a decline is restored
// when the cell is back at its target, which already holds what its readings allowed, whatever spread
// it carries. A throughput that fell from 100 to 80 after five readings of 100 is filed with 100 as its
// target, and is filed under the same key for as long as it stays at 80 and 100 is a level among its
// readings: once eight readings in a row are 80, 80 is what it holds and nothing is filed.
func TestACellsOwnSpreadNeitherHidesItsDeclineNorRestoresIt(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "noisy-model")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "noisy-environment")
	noisy, stepped := testutil.ArtifactID(t, artifact.KindRecipe, "noisy-recipe"), testutil.ArtifactID(t, artifact.KindRecipe, "stepped-recipe")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "noisy/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: environment}, {ID: noisy}, {ID: stepped}}}); err != nil {
		t.Fatal(err)
	}
	const millisecond = uint64(time.Millisecond)
	// Committed oldest first. Four attempts of the noisy recipe loaded their model; the oldest read 9 ms.
	// Three of the stepped recipe did, and the newest is the slow one.
	for served, prefills := range map[artifact.ID][]uint64{noisy: {9, 16, 23, 49}, stepped: {100, 100, 160}} {
		for index, prefill := range prefills {
			started := int64(index + 1)
			observation, err := runrecord.NewServingObservation(runrecord.ServingObservation{
				Model: model, Recipe: served, Environment: environment, Task: recipe.TaskInference,
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
	}
	summaries, err := servingByRecipe(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	cell := firstOutputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, summaries[noisy])
	if want := 23.0; cell.Value != want || math.Abs(cell.Spread-7.0/23) > 1e-9 {
		t.Fatalf("first output = %v with spread %v, want %v, the median of 16, 23 and 49 ms, and 7/23, their median distance from it over it", cell.Value, cell.Spread, want)
	}
	if cell := firstOutputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, summaries[stepped]); cell.Value != 100 || cell.Spread != 0 {
		t.Fatalf("first output = %v with spread %v, want the median 100 and no spread: one slow attempt of three moves neither", cell.Value, cell.Spread)
	}

	contract := Contract{Metric: "time to prefill", Origin: originServing}
	timing := func(value, spread, allowed float64) Cell {
		return Cell{Value: value, Spread: spread, Unit: "ms", Contract: contract, allowed: allowed}
	}
	for _, test := range []struct {
		name      string
		then, now Cell
		want      Verdict
	}{
		{"a stable cell at 160 ms from 100, its own three attempts spread by a half", timing(100, 0.02, 25), timing(160, 0.5, 0), regressed},
		{"a 9 ms operation at 30 ms, its repeats spread by 1.4", timing(9, 0, 2), timing(30, 1.43, 0), regressed},
		{"the same cell from a reference that publishes that spread", timing(9, 1.43, 2), timing(30, 0, 0), regressed},
		{"the same cell from a reference whose readings allow 25 ms", timing(9, 0, 25), timing(30, 0, 0), holding},
		{"a stable cell at 160 ms from 100", timing(100, 0.02, 25), timing(160, 0.02, 0), regressed},
		{"a stable cell at 120 ms from 100", timing(100, 0.02, 25), timing(120, 0.02, 0), holding},
	} {
		if verdict, why := holds(metric.FirstOutput, test.then, test.now); verdict != test.want {
			t.Errorf("%s: holds = %q (%s), want %q", test.name, verdict, why, test.want)
		}
	}
	for _, test := range []struct {
		name     string
		cell     Cell
		target   float64
		restored bool
	}{
		{"the stepped cell at 160 ms against its target of 125", timing(160, 0.5, 0), 125, false},
		{"the noisy cell at 30 ms against its target of 11", timing(30, 1.43, 0), 11, false},
		{"a cell at 120 ms against its target of 125", timing(120, 0, 0), 125, true},
	} {
		row := Row{Model: model, Task: recipe.TaskInference, Cells: map[metric.Name]Cell{metric.FirstOutput: test.cell}}
		key := regressionKey(row, metric.FirstOutput, test.cell)
		if restored, found := Restored([]Row{row}, key, test.target); restored != test.restored || !found {
			t.Errorf("%s = (restored %t, found %t), want (restored %t, found): a decline is restored by reaching its target, never by the cell's own spread", test.name, restored, found, test.restored)
		}
	}

	// A throughput that read 100 tok/s five times, fell to 80 and stayed there. Each record's cell carries
	// a spread set by hand, the range of its newest three over their mean, which judges nothing.
	rate := func(value, spread float64) Record {
		throughput := Cell{Value: value, Spread: spread, Unit: "tok/s", Contract: Contract{Metric: "output tokens over decode time", Origin: originServing}}
		return Record{Rows: []Row{measuredFixture(model, map[metric.Name]Cell{metric.Throughput: throughput})}}
	}
	history := slices.Repeat([]Record{rate(100, 0)}, fewestReadings)
	fell, stayed := rate(80, 60.0/280), rate(80, 60.0/260)
	declines := func(records ...Record) []Gap {
		newest := records[len(records)-1]
		return slices.DeleteFunc(newest.Gaps(Reference(records)), func(gap Gap) bool { return gap.kind != string(regressed) })
	}
	key := regressionKey(fell.Rows[0], metric.Throughput, fell.Rows[0].Cells[metric.Throughput])
	if filed := declines(append(slices.Clone(history), fell)...); len(filed) != 1 || filed[0].Key != key || filed[0].Target != 100 || !filed[0].Restore {
		t.Errorf("declines of a throughput that fell from 100 to 80 = %+v, want the one, to be restored to 100", filed)
	}
	if few := declines(append(slices.Clone(history[1:]), fell)...); len(few) != 0 {
		t.Errorf("declines of a throughput that fell to 80 after four readings of 100 = %+v, want none: four readings do not say what the cell holds", few)
	}
	// While 100 is a level among the newest nine readings the cell is still declined, under the same key.
	for standing := 1; standing < referenceWindow-1; standing++ {
		records := append(append(slices.Clone(history), slices.Repeat([]Record{fell}, standing)...), stayed)
		if again := declines(records...); len(again) != 1 || again[0].Key != key || again[0].Target != 100 {
			t.Errorf("declines of a throughput at 80 for %d readings after five of 100 = %+v, want the same one, to be restored to 100", standing+1, again)
		}
	}
	settled := append(append(slices.Clone(history), slices.Repeat([]Record{fell}, referenceWindow-1)...), stayed)
	if again := declines(settled...); len(again) != 0 {
		t.Errorf("declines of a throughput at 80 for nine readings = %+v, want none: 100 has left the readings it is judged by, and 80 is what it holds", again)
	}
	if restored, found := Restored(stayed.Rows, key, 100); restored || !found {
		t.Errorf("a throughput still at 80 against its target of 100 = (restored %t, found %t), want found and not restored", restored, found)
	}
}
