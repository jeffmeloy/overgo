package main

import (
	"slices"
	"strings"
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
// moved since the first, rows paired by model and task: a changed value with
// its delta, under the model's new name; a cell that gained its measure; a
// cell that lost it; a first output in the same unit from another kind of
// record, which does not compare; a new row and a gone row, though the two
// share a name.
func TestScheduledScorecardPublishesDeltas(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	source := testutil.ArtifactID(t, artifact.KindEvidence, "scorecard-deltas")
	comparison := testutil.ArtifactID(t, artifact.KindProfile, "benchmark-execution")
	decode := scorecard.Contract{Metric: "decode rate, median", Origin: "benchmark", Comparison: comparison}
	row := func(model, name string, cells map[metric.Name]scorecard.Cell) scorecard.Row {
		for _, column := range metric.Columns {
			if _, set := cells[column]; !set {
				cells[column] = scorecard.Cell{Reason: "not measured in this fixture"}
			}
		}
		return scorecard.Row{Model: testutil.ArtifactID(t, artifact.KindModel, model), Name: name, Task: recipe.TaskInference, Cells: cells}
	}
	before := scorecard.Record{Commit: "a", Rows: []scorecard.Row{
		row("alpha", "alpha", map[metric.Name]scorecard.Cell{
			metric.Throughput:  {Value: 40, Unit: "tok/s", Source: source, Contract: decode},
			metric.Quality:     {Reason: "no held-out record"},
			metric.PeakMemory:  {Value: 8, Unit: "bytes", Source: source, Contract: scorecard.Contract{Metric: "peak device memory", Origin: "benchmark", Comparison: comparison}},
			metric.FirstOutput: {Value: 120, Unit: "ms", Source: source, Contract: scorecard.Contract{Metric: "time to first token, median", Origin: "benchmark", Comparison: comparison}},
		}),
		row("retired", "beta", map[metric.Name]scorecard.Cell{metric.Throughput: {Value: 9, Unit: "tok/s", Source: source, Contract: decode}}),
	}}
	after := scorecard.Record{Commit: "b", Rows: []scorecard.Row{
		row("alpha", "alpha-renamed", map[metric.Name]scorecard.Cell{
			metric.Throughput:  {Value: 45, Unit: "tok/s", Source: source, Contract: decode},
			metric.Quality:     {Value: 0.7, Unit: "accuracy", Source: source},
			metric.PeakMemory:  {Reason: "no benchmark record"},
			metric.FirstOutput: {Value: 130, Unit: "ms", Source: source, Contract: scorecard.Contract{Metric: "time to prefill", Origin: "serving"}},
		}),
		row("beta", "beta", map[metric.Name]scorecard.Cell{metric.Throughput: {Value: 30, Unit: "tok/s", Source: source, Contract: decode}}),
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
		"- alpha-renamed inference, Throughput: 40 to 45 tok/s (+5)",
		"- alpha-renamed inference, Held-out quality: now measured, 0.7 accuracy",
		"- alpha-renamed inference, Peak memory: no current value: no benchmark record",
		"- beta inference: new",
		"- beta inference: gone",
	}
	moved := scorecardMoves(records[0], records[1])
	// The user needs both incompatible metrics and origins, named in words.
	if len(moved) != len(want)+1 || !strings.HasPrefix(moved[0], "- alpha-renamed inference, First output: measured on another contract: ") ||
		!strings.Contains(moved[0], `the measure is "time to prefill", the reference's "time to first token, median"`) ||
		!strings.Contains(moved[0], `the origin is "serving", the reference's "benchmark"`) ||
		!slices.Equal(moved[1:], want) {
		t.Fatalf("moved =\n%q\nwant\n%q", moved, want)
	}
	legacy := records[0].Rows[0].Cells[metric.Throughput]
	legacy.Contract.Comparison = artifact.ID{}
	records[0].Rows[0].Cells[metric.Throughput] = legacy
	if moved := scorecardMoves(records[0], records[1]); !slices.Contains(moved, "- alpha-renamed inference, Throughput: benchmark comparison authority is unknown") {
		t.Fatalf("legacy benchmark reports %q, want an unknown comparison instead of a numerical delta", moved)
	}
}
