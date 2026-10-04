package scorecard

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestMovesBetweenRecords compares two published records through the
// promotion's comparison: a value that changed on one contract, a cell that
// gained or lost its measure, a cell measured on another contract (Unknown,
// never better or worse), a row that came and one that went; a rising
// throughput improved and a rising cost did not, a held-out quality by its
// own direction; and rows pair by model, so two sharing a name never trade
// baselines.
func TestMovesBetweenRecords(t *testing.T) {
	t.Parallel()
	model := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindModel, name) }
	cell := func(column metric.Name, value float64, unit string) Cell {
		return Cell{Value: value, Unit: unit, Contract: Contract{Metric: string(column), Origin: originServing}}
	}
	// A published record carries a cell for every column, a gap naming its reason.
	row := func(id artifact.ID, name string, cells map[metric.Name]Cell) Row {
		for _, column := range metric.Columns {
			if _, found := cells[column]; !found {
				cells[column] = Cell{Reason: "no record measures it"}
			}
		}
		return Row{Model: id, Name: name, Task: recipe.TaskInference, Cells: cells}
	}
	quality := func(value float64, suite string) Cell {
		return Cell{Value: value, Unit: "accuracy", Direction: runrecord.DirectionMinimize,
			Contract: Contract{Metric: "perplexity", Origin: originServing, Suite: suite}}
	}
	alpha, twin, retired := model("alpha"), model("twin"), model("retired")
	before := Record{Commit: "a", Rows: []Row{
		row(alpha, "alpha", map[metric.Name]Cell{
			metric.Throughput: cell(metric.Throughput, 40, "tok/s"), metric.FirstOutput: cell(metric.FirstOutput, 100, "ms"),
			metric.Quality: quality(12, "wikitext"), metric.PeakMemory: {Reason: "no benchmark record"}, metric.Load: cell(metric.Load, 5, "ms"),
		}),
		row(twin, "same name", map[metric.Name]Cell{metric.Throughput: cell(metric.Throughput, 10, "tok/s")}),
		row(retired, "retired", map[metric.Name]Cell{metric.Throughput: cell(metric.Throughput, 9, "tok/s")}),
	}}
	elsewhere := cell(metric.Load, 6, "ms")
	elsewhere.Contract.Scope = model("other recipe")
	after := Record{Commit: "b", Rows: []Row{
		row(alpha, "alpha", map[metric.Name]Cell{
			metric.Throughput: cell(metric.Throughput, 45, "tok/s"), metric.FirstOutput: cell(metric.FirstOutput, 130, "ms"),
			metric.Quality: quality(10, "wikitext"), metric.PeakMemory: cell(metric.PeakMemory, 8, "bytes"), metric.Load: elsewhere,
		}),
		row(twin, "same name", map[metric.Name]Cell{metric.Throughput: cell(metric.Throughput, 10, "tok/s")}),
		row(model("beta"), "same name", map[metric.Name]Cell{metric.Throughput: cell(metric.Throughput, 30, "tok/s")}),
	}}
	moves := Moves(before, after)
	var got []string
	for _, move := range moves {
		got = append(got, move.Name+" "+string(move.Column)+" "+string(move.Kind))
	}
	want := []string{"alpha first-output changed", "alpha throughput changed", "alpha held-out-quality changed",
		"alpha load-time unknown", "alpha peak-device-memory measured", "same name  added", "retired  removed"}
	if !slices.Equal(got, want) {
		t.Fatalf("moves = %q, want %q", got, want)
	}
	if moves[0].Advantage() >= 0 || moves[1].Advantage() <= 0 || moves[2].Advantage() <= 0 {
		t.Fatalf("a slower first output improved, a faster throughput or a lower perplexity did not: %+v", moves[:3])
	}
	if moves[3].Reason == "" || moves[3].Advantage() != 0 {
		t.Fatalf("a cell measured on another contract moved as better or worse: %+v", moves[3])
	}
	if none := Moves(after, after); len(none) != 0 {
		t.Fatalf("a record moved against itself: %+v", none)
	}
}
