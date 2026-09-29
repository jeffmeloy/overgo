package main

import (
	"slices"
	"testing"

	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
)

// TestScheduledScorecardPublishesDeltas publishes two scorecards into one
// directory and holds the second to what moved since the first: a changed
// value with its delta, a cell that gained its measure, a cell that lost it,
// a new row and a gone row. The first publish has nothing to compare with.
func TestScheduledScorecardPublishesDeltas(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	row := func(name string, cells map[metric.Name]scorecard.Cell) scorecard.Row {
		return scorecard.Row{Name: name, Task: recipe.TaskInference, Cells: cells}
	}
	before := []scorecard.Row{
		row("alpha", map[metric.Name]scorecard.Cell{
			metric.Throughput:  {Value: 40, Unit: "tok/s"},
			metric.Quality:     {Reason: "no held-out record"},
			metric.PeakMemory:  {Value: 8, Unit: "bytes"},
			metric.FirstOutput: {Value: 120, Unit: "ms"},
		}),
		row("retired", map[metric.Name]scorecard.Cell{metric.Throughput: {Value: 9, Unit: "tok/s"}}),
	}
	if moved, first, err := publishScorecard(directory, "scorecard-20260928T100000Z.json", "a", before); err != nil || !first || len(moved) != 0 {
		t.Fatalf("first publish = (%v, first %v, %v), want no comparison", moved, first, err)
	}
	after := []scorecard.Row{
		row("alpha", map[metric.Name]scorecard.Cell{
			metric.Throughput:  {Value: 45, Unit: "tok/s"},
			metric.Quality:     {Value: 0.7, Unit: "accuracy"},
			metric.PeakMemory:  {Reason: "no benchmark record"},
			metric.FirstOutput: {Value: 120, Unit: "ms"},
		}),
		row("beta", map[metric.Name]scorecard.Cell{metric.Throughput: {Value: 30, Unit: "tok/s"}}),
	}
	moved, first, err := publishScorecard(directory, "scorecard-20260928T110000Z.json", "b", after)
	if err != nil || first {
		t.Fatalf("second publish = (first %v, %v)", first, err)
	}
	want := []string{
		"- alpha inference, Throughput: 40 to 45 tok/s (+5)",
		"- alpha inference, Held-out quality: now measured, 0.7 accuracy",
		"- alpha inference, Peak memory: no longer measured: no benchmark record",
		"- beta inference: new",
		"- retired inference: gone",
	}
	if !slices.Equal(moved, want) {
		t.Fatalf("moved =\n%q\nwant\n%q", moved, want)
	}
	latest, err := scorecard.LatestRecords(directory, 1)
	if err != nil || len(latest) != 1 || latest[0].Commit != "b" {
		t.Fatalf("latest record = (%+v, %v), want the second publish", latest, err)
	}
}
