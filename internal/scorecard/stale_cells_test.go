package scorecard

import (
	"maps"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
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

// TestATrainedModelsReadingStands: a trained model is measured once, so a later record keeps its reading, where any other
// model's reading from before the previous record says it was not measured.
func TestATrainedModelsReadingStands(t *testing.T) {
	t.Parallel()
	published := time.Date(2026, 11, 2, 9, 0, 0, 0, time.UTC)
	reading := Cell{Value: 1, Newest: published.Add(-time.Hour).UnixNano()}
	row := func(once bool) Row {
		return Row{MeasuredOnce: once, Cells: map[metric.Name]Cell{metric.Throughput: reading}}
	}
	got := Record{Rows: []Row{row(true), row(false)}}.Unmeasured(published)
	if cell := got.Rows[0].Cells[metric.Throughput]; cell.Value != reading.Value || cell.Reason != "" {
		t.Fatalf("the trained model's reading became %+v", cell)
	}
	if cell := got.Rows[1].Cells[metric.Throughput]; cell.Reason == "" {
		t.Fatalf("another model's old reading stands as %+v", cell)
	}
}

// TestAQualityCellNotMeasuredThisRunSaysSo: a held-out quality whose evaluation failed this run is a gap that names what
// happened and the reading before it, and is no longer compared, where the older value read as holding. A model's other
// suite and another model's reading of the suite stand, and a quality no evaluation failed stands however old it is.
func TestAQualityCellNotMeasuredThisRunSaysSo(t *testing.T) {
	t.Parallel()
	failing := testutil.ArtifactID(t, artifact.KindModel, "failing")
	other := testutil.ArtifactID(t, artifact.KindModel, "other")
	quality := func(suite string, value float64) Cell {
		return Cell{Value: value, Source: testutil.ArtifactID(t, artifact.KindEvaluation, suite), Direction: runrecord.DirectionMaximize,
			Contract: Contract{Metric: shareMetric, Origin: originEvaluation, Suite: suite, Comparison: testutil.ArtifactID(t, artifact.KindProfile, suite)}}
	}
	row := func(model artifact.ID) Row {
		return Row{
			Model: model, Name: "paired", Task: recipe.TaskInference, Cells: map[metric.Name]Cell{metric.Quality: quality("store/mmlu-pro", 0.5)},
			Heldout: []Cell{quality("store/bbh", 0.25), quality("store/gsm8k", 0.75)},
		}
	}
	record := Record{Rows: []Row{row(failing), row(other)}}
	failed := func(model artifact.ID, suite string) bool {
		return model == failing && (suite == "store/mmlu-pro" || suite == "store/bbh")
	}
	got := record.NotEvaluated(failed)
	cell := got.Rows[0].Cells[metric.Quality]
	if cell.Measured() || !strings.Contains(cell.Reason, "its evaluation on store/mmlu-pro failed") || !strings.Contains(cell.Reason, "was 0.5") {
		t.Fatalf("the failed evaluation's cell reads %+v", cell)
	}
	if kept := got.Rows[0].Heldout; len(kept) != 1 || kept[0].Contract.Suite != "store/gsm8k" {
		t.Fatalf("the model's other suites read %+v, want the one whose evaluation did not fail", kept)
	}
	if cell := got.Rows[1].Cells[metric.Quality]; !cell.Measured() || len(got.Rows[1].Heldout) != 2 {
		t.Fatalf("another model's readings became %+v and %+v", cell, got.Rows[1].Heldout)
	}
	if !record.Rows[0].Cells[metric.Quality].Measured() || len(record.Rows[0].Heldout) != 2 {
		t.Fatal("the record it was read from changed")
	}
	for _, comparison := range (Pair{Before: &record.Rows[0], After: &got.Rows[0]}).Compare(metric.Quality) {
		if comparison.Verdict == holding || comparison.Verdict == regressed {
			t.Fatalf("a quality this run did not measure compares as %+v", comparison)
		}
	}
}
