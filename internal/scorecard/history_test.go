package scorecard

import (
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/scorecard/metric"
)

// TestMovesBetweenRecords compares two published records: a value that
// changed, a cell that gained or lost its measure, a row that came and one
// that went; a rising throughput or quality improved, a rising cost did not;
// and the directory's newest records read back oldest first.
func TestMovesBetweenRecords(t *testing.T) {
	t.Parallel()
	row := func(name string, cells map[metric.Name]RecordedCell) RecordedRow {
		return RecordedRow{Name: name, Task: "inference", Cells: cells}
	}
	before := Record{Commit: "a", Rows: []RecordedRow{
		row("alpha", map[metric.Name]RecordedCell{
			metric.Throughput: {Value: 40, Unit: "tok/s"}, metric.FirstOutput: {Value: 100, Unit: "ms"},
			metric.Quality: {Reason: "no held-out record"}, metric.PeakMemory: {Value: 8, Unit: "bytes"},
		}),
		row("retired", map[metric.Name]RecordedCell{metric.Throughput: {Value: 9, Unit: "tok/s"}}),
	}}
	after := Record{Commit: "b", Rows: []RecordedRow{
		row("alpha", map[metric.Name]RecordedCell{
			metric.Throughput: {Value: 45, Unit: "tok/s"}, metric.FirstOutput: {Value: 130, Unit: "ms"},
			metric.Quality: {Value: 0.7, Unit: "accuracy"}, metric.PeakMemory: {Reason: "no benchmark record"},
		}),
		row("beta", map[metric.Name]RecordedCell{metric.Throughput: {Value: 30, Unit: "tok/s"}}),
	}}
	moves := Moves(before, after)
	kinds := []string{}
	for _, move := range moves {
		kinds = append(kinds, move.Name+" "+string(move.Column)+" "+string(move.Kind))
	}
	want := []string{"alpha first-output changed", "alpha throughput changed", "alpha held-out-quality measured",
		"alpha peak-device-memory unmeasured", "beta  added", "retired  removed"}
	if !slices.Equal(kinds, want) {
		t.Fatalf("moves = %q, want %q", kinds, want)
	}
	if moves[0].Improved() || !moves[1].Improved() || moves[2].Improved() {
		t.Fatalf("a slower first output improved or a faster throughput did not: %+v", moves[:3])
	}

	directory := t.TempDir()
	for index, record := range []Record{before, after, {Commit: "c"}} {
		if err := jsonfile.Write(filepath.Join(directory, RecordName("20260929T00000"+string(rune('0'+index))+"Z")), record, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	latest, err := LatestRecords(directory, 2)
	if err != nil || len(latest) != 2 || latest[0].Commit != "b" || latest[1].Commit != "c" {
		t.Fatalf("the newest two records = %+v, %v", latest, err)
	}
	if none, err := LatestRecords(filepath.Join(directory, "absent"), 2); err != nil || len(none) != 0 {
		t.Fatalf("an absent directory read %+v, %v", none, err)
	}
}
