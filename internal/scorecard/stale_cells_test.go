package scorecard

import (
	"maps"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestACellNotMeasuredThisRunSaysSo holds a published serving cell to the run
// that measured it. A timing cell names when the newest attempt of its window
// started; a cell whose newest attempt started before the previous record was
// published, as Qwen3.8 Flash Next's would had this run's attempt failed and
// its window kept the run before's, is a gap that says when it was last
// measured, never a reading, and a cell measured since stands. A longer
// prompt's or a warm request's cell not measured since is left out. When a
// record was published is read back from the name Publish gives its file.
func TestACellNotMeasuredThisRunSaysSo(t *testing.T) {
	t.Parallel()
	published := time.Date(2026, 10, 30, 9, 44, 20, 0, time.UTC)
	before, after := published.Add(-time.Hour).UnixNano(), published.Add(time.Hour).UnixNano()
	attempt := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindEvidence, name) }

	explained := Row{Model: testutil.ArtifactID(t, artifact.KindModel, "flash"), Name: "Qwen3.8 Flash Next", Task: recipe.TaskInference, Cells: map[metric.Name]Cell{}}
	for _, column := range metric.Columns {
		explained.Cells[column] = gap("not measured in this fixture")
	}
	directory := t.TempDir()
	if _, err := Publish(directory, published, Record{Commit: "a", Rows: []Row{explained}}); err != nil {
		t.Fatal(err)
	}
	records, err := Records(directory)
	if err != nil || len(records) != 1 {
		t.Fatalf("read %d records (%v), want the one published", len(records), err)
	}
	since, found := records[0].Published()
	if !found || !since.Equal(published) {
		t.Fatalf("the record reads as published %v (%t), want %v", since, found, published)
	}
	if _, found := (Record{}).Published(); found {
		t.Fatal("a record read from no file says when it was published")
	}

	window := &servingSummary{window: []servingSample{
		{attempt: attempt("newer"), started: after, took: [timingCount]float64{timingLoad: 31675}},
		{attempt: attempt("older"), started: before, took: [timingCount]float64{timingLoad: 30325}},
	}}
	if cell, measured := window.timed(timingLoad, "ms", "load phase", "load phase"); !measured || cell.Newest != after {
		t.Fatalf("a timing read from a window names its newest attempt's start %d (%t), want %d", cell.Newest, measured, after)
	}

	row := explained
	row.Cells = maps.Clone(explained.Cells)
	row.Cells[metric.FirstOutput] = Cell{Value: 2803.8, Unit: "ms", Source: attempt("older"), Newest: before}
	row.Cells[metric.Load] = Cell{Value: 31675, Unit: "ms", Source: attempt("newer"), Newest: after}
	row.Lengths = []Cell{{Value: 2944.9, Unit: "ms", Source: attempt("older"), Prompt: 4096, Newest: before}}
	row.Warm = []Cell{{Value: 86.79, Unit: "tok/s", Source: attempt("newer"), Contract: Contract{Metric: warmRateMetric}, Newest: after}}
	record := Record{Commit: "b", Rows: []Row{row}}
	judged := record.Unmeasured(since).Rows[0]
	if first := judged.Cells[metric.FirstOutput]; first.Measured() || !strings.Contains(first.Reason, "not measured since the previous scorecard") {
		t.Fatalf("a cell whose newest attempt predates the previous record reads %+v, want the gap that says so", first)
	}
	if judged.Cells[metric.Load].Value != 31675 || len(judged.Lengths) != 0 || len(judged.Warm) != 1 {
		t.Fatalf("cells measured since stand and those not are left out: load %+v, lengths %+v, warm %+v", judged.Cells[metric.Load], judged.Lengths, judged.Warm)
	}
	if record.Rows[0].Cells[metric.FirstOutput].Value != 2803.8 || len(record.Rows[0].Lengths) != 1 {
		t.Fatal("judging the record changed the record it judged")
	}
	gaps := (Record{Rows: []Row{judged}}).Gaps(nil)
	if !slices.ContainsFunc(gaps, func(gap Gap) bool {
		return gap.Key == gapKey(gapUnmeasured, recipe.TaskInference) && strings.Contains(gap.Detail, "not measured since the previous scorecard")
	}) {
		t.Fatalf("the cell not measured since is no gap: %+v", gaps)
	}
}
