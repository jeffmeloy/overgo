package main

import (
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestScheduledScorecardPublishesDeltas publishes two scorecards into one
// directory, reads them back oldest first, and holds the second to what
// moved since the first: a changed value with its delta, a cell that gained
// its measure, a cell that lost it, a new row and a gone row.
func TestScheduledScorecardPublishesDeltas(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	source := testutil.ArtifactID(t, artifact.KindEvidence, "scorecard-deltas")
	row := func(name string, cells map[metric.Name]scorecard.Cell) scorecard.Row {
		for _, column := range metric.Columns {
			if _, set := cells[column]; !set {
				cells[column] = scorecard.Cell{Reason: "not measured in this fixture"}
			}
		}
		return scorecard.Row{Name: name, Task: recipe.TaskInference, Cells: cells}
	}
	before := scorecard.Record{Commit: "a", Rows: []scorecard.Row{
		row("alpha", map[metric.Name]scorecard.Cell{
			metric.Throughput:  {Value: 40, Unit: "tok/s", Source: source},
			metric.Quality:     {Reason: "no held-out record"},
			metric.PeakMemory:  {Value: 8, Unit: "bytes", Source: source},
			metric.FirstOutput: {Value: 120, Unit: "ms", Source: source},
		}),
		row("retired", map[metric.Name]scorecard.Cell{metric.Throughput: {Value: 9, Unit: "tok/s", Source: source}}),
	}}
	after := scorecard.Record{Commit: "b", Rows: []scorecard.Row{
		row("alpha", map[metric.Name]scorecard.Cell{
			metric.Throughput:  {Value: 45, Unit: "tok/s", Source: source},
			metric.Quality:     {Value: 0.7, Unit: "accuracy", Source: source},
			metric.PeakMemory:  {Reason: "no benchmark record"},
			metric.FirstOutput: {Value: 120, Unit: "ms", Source: source},
		}),
		row("beta", map[metric.Name]scorecard.Cell{metric.Throughput: {Value: 30, Unit: "tok/s", Source: source}}),
	}}
	published := time.Date(2026, time.September, 28, 10, 0, 0, 0, time.UTC)
	for index, record := range []scorecard.Record{before, after} {
		if _, err := scorecard.Publish(directory, published.Add(time.Duration(index)*time.Hour), record); err != nil {
			t.Fatal(err)
		}
	}
	records, err := scorecard.Records(directory)
	if err != nil || len(records) != 2 || records[0].Commit != "a" || records[1].Commit != "b" {
		t.Fatalf("records = %+v, %v; want the two publishes, oldest first", records, err)
	}
	want := []string{
		"- alpha inference, Throughput: 40 to 45 tok/s (+5)",
		"- alpha inference, Held-out quality: now measured, 0.7 accuracy",
		"- alpha inference, Peak memory: no longer measured: no benchmark record",
		"- beta inference: new",
		"- retired inference: gone",
	}
	if moved := scorecardMoves(records[0], records[1]); !slices.Equal(moved, want) {
		t.Fatalf("moved =\n%q\nwant\n%q", moved, want)
	}
}
