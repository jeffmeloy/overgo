package scorecard

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestScorecardUsesCompatibleFixedReferencesAndMetricSpecificEvidence holds what a record's cell is
// judged against to the cell's own earlier readings, by evidence of the cell's kind.
//
// A load time read 100, 101, 99, 100 and 101 ms and then 104, 109 and 114: three declines of about
// 4%, each within any allowance of the reading before it. The first holds the 100 ms its readings held
// and the 8 ms eight of their steps allow. The second and third do not, and are filed under the one key,
// the second with a target of 108 ms, at which the cell is restored. Three declines of 30% each, the case that
// passed before, are caught at the first.
//
// Two cells of one column at 110 ms from 100: the one whose readings step by half a millisecond is a
// decline and the one whose readings step by 25 is not. A cell with four earlier readings is Unknown
// at three times its level, and is never filed and never holding. An improvement held twice is the
// level, so a reading that gives half of it back is a decline, and one fast reading alone is none. A
// value several records carry from one source is one reading, and a reading on another recipe or under
// another environment is another cell's.
//
// A held-out quality, a count of setup steps and a failed share need no repeats: one earlier record
// judges each, the first two by any move the printed digits show, the third by its attempts.
func TestScorecardUsesCompatibleFixedReferencesAndMetricSpecificEvidence(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "judged")
	load := Contract{Metric: "load", Origin: originServing, Scope: testutil.ArtifactID(t, artifact.KindRecipe, "judged-recipe")}
	read := func(column metric.Name, cell Cell) Record {
		return Record{Rows: []Row{measuredFixture(model, map[metric.Name]Cell{column: cell})}}
	}
	timed := func(values ...float64) []Record {
		records := make([]Record, len(values))
		for index, value := range values {
			records[index] = read(metric.Load, Cell{Value: value, Unit: "ms", Contract: load})
		}
		return records
	}
	// judged is how the newest of records compares with the reference the records before it give.
	judged := func(column metric.Name, records []Record) (declines []Gap, compared []Comparison) {
		newest, reference := records[len(records)-1], Reference(records)
		for _, pair := range newest.Pairs(*reference) {
			compared = append(compared, pair.Compare(column)...)
		}
		return slices.DeleteFunc(newest.Gaps(reference), func(gap Gap) bool { return gap.kind != string(regressed) }), compared
	}

	steady := []float64{100, 101, 99, 100, 101}
	if filed, compared := judged(metric.Load, timed(append(slices.Clone(steady), 104)...)); len(filed) != 0 || len(compared) != 0 {
		t.Errorf("104 ms after readings of %v = %+v and %+v, want it to hold: it is within the 8 ms eight of their steps allow of the 100 they held", steady, filed, compared)
	}
	// The third is judged by readings that hold the decline's own first two steps, which widen what is
	// allowed from 8 to 12 ms. The row filed at the second keeps its target of 108.
	var key string
	for _, crept := range []struct {
		readings []float64
		target   float64
	}{{[]float64{104, 109}, 108}, {[]float64{104, 109, 114}, 112}} {
		filed, _ := judged(metric.Load, timed(append(slices.Clone(steady), crept.readings...)...))
		if len(filed) != 1 || filed[0].Target != crept.target || !filed[0].Restore || !strings.Contains(filed[0].Title, fmt.Sprintf("to %v ms", crept.target)) ||
			!strings.Contains(filed[0].Detail, fmt.Sprintf("allowed %v;", crept.target-100)) {
			t.Fatalf("declines after readings of %v then %v = %+v, want one, to be restored to the %v ms that holds the level of 100", steady, crept.readings, filed, crept.target)
		}
		if key != "" && filed[0].Key != key {
			t.Errorf("the decline at %v is filed as %s, and the one before it as %s: a cell that goes further down keeps its key", crept.readings, filed[0].Key, key)
		}
		key = filed[0].Key
	}
	for ms, want := range map[float64]bool{108: true, 109: false} {
		rows := read(metric.Load, Cell{Value: ms, Unit: "ms", Contract: load}).Rows
		if restored, found := Restored(rows, key, 108); restored != want || !found {
			t.Errorf("the cell at %v ms against its target of 108 = (restored %t, found %t), want restored %t", ms, restored, found, want)
		}
	}
	if filed, _ := judged(metric.Load, timed(append(slices.Clone(steady), 130)...)); len(filed) != 1 {
		t.Errorf("declines at 130 ms after readings of %v = %+v, want one: the first of three declines of 30%% is a decline", steady, filed)
	}

	// The same 10 ms beyond 100, against readings that step by half a millisecond and by 25.
	if filed, _ := judged(metric.Load, timed(100, 100.5, 100, 100.5, 100, 110)); len(filed) != 1 || filed[0].Target != 104 {
		t.Errorf("declines at 110 ms after readings that step by 0.5 ms = %+v, want one, to be restored to 104", filed)
	}
	if filed, compared := judged(metric.Load, timed(100, 120, 95, 125, 100, 110)); len(filed) != 0 || len(compared) != 0 {
		t.Errorf("110 ms after readings that step by 25 ms = %+v and %+v, want it to hold: its readings allow 200 ms beyond the 100 they held", filed, compared)
	}

	// Four earlier readings say neither what the cell holds nor how far it moves.
	filed, compared := judged(metric.Load, timed(100, 101, 99, 100, 300))
	if len(filed) != 0 || len(compared) != 1 || compared[0].Verdict != Unknown || !strings.Contains(compared[0].Why, fmt.Sprintf("%d earlier readings on its contract, fewer than the %d", fewestReadings-1, fewestReadings)) {
		t.Errorf("300 ms after four readings near 100 = %+v and %+v, want no decline and one comparison that is unknown for its too few readings", filed, compared)
	}

	// An improvement is the level once two readings hold it.
	if filed, _ := judged(metric.Load, timed(100, 100, 100, 100, 100, 50, 50, 70)); len(filed) != 1 || filed[0].Target != 50 {
		t.Errorf("declines at 70 ms after five readings of 100 and two of 50 = %+v, want one, to be restored to 50", filed)
	}
	if filed, compared := judged(metric.Load, timed(100, 100, 100, 100, 100, 50, 100)); len(filed) != 0 || len(compared) != 0 {
		t.Errorf("100 ms after five readings of 100 and one of 50 = %+v and %+v, want it to hold: one fast reading is no level", filed, compared)
	}

	// Nine records carry three readings, each from its own source: too few, however many records.
	var carried []Record
	for index, value := range []float64{100, 101, 99} {
		cell := Cell{Value: value, Unit: "ms", Contract: load, Source: testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("attempt-%d", index))}
		carried = append(carried, slices.Repeat([]Record{read(metric.Load, cell)}, levelReadings)...)
	}
	if filed, compared := judged(metric.Load, append(carried, timed(300)...)); len(filed) != 0 || len(compared) != 1 || !strings.Contains(compared[0].Why, "3 earlier readings") {
		t.Errorf("300 ms after nine records that carry three readings = %+v and %+v, want it unknown for three readings", filed, compared)
	}

	// Nonadjacent publications of the same source and value remain one reading.
	interleaved := []Record{carried[0], carried[3], carried[6], carried[0], carried[3], carried[6]}
	if filed, compared := judged(metric.Load, append(interleaved, timed(300)...)); len(filed) != 0 || len(compared) != 1 || !strings.Contains(compared[0].Why, "3 earlier readings") {
		t.Errorf("interleaved carried readings = %+v and %+v, want unknown for three distinct readings", filed, compared)
	}
	// The newest prior contract must not hide enough older readings on the current one.
	compatible := timed(100, 100, 100, 100, 100)
	compatible = append(compatible, read(metric.Load, Cell{Value: 5, Contract: Contract{Metric: "different"}}), timed(300)[0])
	if filed, compared := judged(metric.Load, compatible); len(filed) != 1 || len(compared) != 1 || compared[0].Verdict != regressed {
		t.Errorf("compatible history behind another contract = %+v and %+v, want the measured decline", filed, compared)
	}

	// Readings on another recipe, or under another environment, are not this cell's.
	other := load
	other.Scope = testutil.ArtifactID(t, artifact.KindRecipe, "another-recipe")
	elsewhere := testutil.ArtifactID(t, artifact.KindEvidence, "another-environment")
	mixed := timed(100, 101, 99)
	mixed = append(mixed, read(metric.Load, Cell{Value: 100, Unit: "ms", Contract: other}), read(metric.Load, Cell{Value: 100, Unit: "ms", Contract: load, Environment: elsewhere}))
	mixed = append(mixed, timed(100, 300)...)
	if filed, compared := judged(metric.Load, mixed); len(filed) != 0 || len(compared) != 1 || !strings.Contains(compared[0].Why, "4 earlier readings") {
		t.Errorf("300 ms after four readings of its own and two of another recipe and environment = %+v and %+v, want it unknown for four readings", filed, compared)
	}

	// The columns no repeat reads differently: one earlier record judges each.
	accuracy := Contract{Metric: "accuracy", Origin: originEvaluation, Suite: "store/mmlu-pro", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "judged-comparison")}
	scored := func(value float64, direction runrecord.Direction) Cell {
		return Cell{Value: value, Direction: direction, Contract: accuracy}
	}
	steps := func(value float64) Cell {
		return Cell{Value: value, Contract: Contract{Metric: "transitions", Origin: originLifecycle}}
	}
	share := func(failures, attempts int) Cell {
		return Cell{Value: float64(failures) / float64(attempts), Attempts: attempts, Contract: Contract{Metric: "failed share", Origin: originServing}}
	}
	for _, test := range []struct {
		name      string
		column    metric.Name
		then, now Cell
		declined  bool
	}{
		{"an accuracy one case in ten thousand lower", metric.Quality, scored(0.2855, runrecord.DirectionMaximize), scored(0.2854, runrecord.DirectionMaximize), true},
		{"an accuracy a summation order lower", metric.Quality, scored(0.2855, runrecord.DirectionMaximize), scored(0.2855*(1-1e-8), runrecord.DirectionMaximize), false},
		{"a perplexity parts in a billion higher", metric.Quality, scored(910.321, runrecord.DirectionMinimize), scored(910.321+6.5e-6, runrecord.DirectionMinimize), false},
		{"one setup step more", metric.SetupSteps, steps(3), steps(4), true},
		{"one more failure in a hundred attempts", metric.Failures, share(1, 100), share(2, 100), false},
		{"two more failures in a hundred attempts", metric.Failures, share(1, 100), share(3, 100), true},
	} {
		filed, compared := judged(test.column, []Record{read(test.column, test.then), read(test.column, test.now)})
		if (len(filed) == 1) != test.declined || len(compared) != len(filed) {
			t.Errorf("%s, against one earlier record = %+v and %+v, want declined %t and nothing unknown", test.name, filed, compared, test.declined)
		}
	}
	if recipe.TaskInference != read(metric.Load, Cell{}).Rows[0].Task {
		t.Fatal("the fixture's row is not the inference row the keys above name")
	}
}
