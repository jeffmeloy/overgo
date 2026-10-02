package scorecard

import (
	"cmp"
	"os"
	"path/filepath"
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

// TestComparisonIsTyped holds the one comparison to identity and typed
// contracts. Rows pair by model and task, never by display name: a renamed
// model keeps its baseline, and two models sharing a name never trade
// theirs. A held-out quality compares only on the baseline's metric, suite,
// protocol, origin and direction, where it holds or regresses. Every
// held-out quality the baseline measured is held, the headline and each
// retained beside it, in whatever order the record lists them: one that
// falls or vanishes while the headline holds withholds promotion. A vanished
// row (every row, too), a lost cell, a changed contract and a baseline cell
// published before contracts are unknown: they withhold promotion and are
// proposed as a gap. A metric, a suite or a row the baseline did not measure
// holds and blocks nothing. Serving aggregates of other recipes' attempts,
// and a benchmark beside a serving median in the same unit, do not compare.
func TestComparisonIsTyped(t *testing.T) {
	t.Parallel()
	source := testutil.ArtifactID(t, artifact.KindEvaluation, "typed-comparison")
	first, second, added := testutil.ArtifactID(t, artifact.KindModel, "first"), testutil.ArtifactID(t, artifact.KindModel, "second"), testutil.ArtifactID(t, artifact.KindModel, "added")
	mmlu := Contract{Metric: "accuracy", Origin: originEvaluation, Suite: "store/mmlu", Protocol: "chat-template"}
	row := func(model artifact.ID, name string, accuracy float64, contract Contract) Row {
		cells := map[metric.Name]Cell{}
		for _, column := range metric.Columns {
			cells[column] = gap("not measured in this fixture")
		}
		cells[metric.Quality] = Cell{Value: accuracy, Label: "accuracy", Source: source, Direction: runrecord.DirectionMaximize, Contract: contract}
		return Row{Model: model, Name: name, Task: recipe.TaskInference, Cells: cells}
	}
	baseline := Record{Commit: "a", Rows: []Row{row(first, "shared", 0.5, mmlu), row(second, "shared", 0.7, mmlu)}}
	record := func(rows ...Row) Record { return Record{Commit: "b", Rows: rows} }
	with := func(change func(*Contract)) Contract {
		contract := mmlu
		change(&contract)
		return contract
	}
	minimized := row(first, "shared", 0.5, mmlu)
	minimized.Cells[metric.Quality] = Cell{Value: 0.5, Source: source, Direction: runrecord.DirectionMinimize, Contract: mmlu}
	lost := row(first, "shared", 0, mmlu)
	lost.Cells[metric.Quality] = inapplicable("the task no longer answers questions")
	newlyMeasured := row(first, "shared", 0.5, mmlu)
	newlyMeasured.Cells[metric.Load] = benchmarkCell(1, "ms", "load", source)
	// A larger suite became the headline; the baseline's suite is still
	// measured beside it, holding or fallen.
	larger := with(func(c *Contract) { c.Suite = "store/mmlu-larger" })
	headlineMoved := func(baselineSuite float64) Row {
		moved := row(first, "shared", 0.1, larger)
		moved.Heldout = []Cell{row(first, "shared", baselineSuite, mmlu).Cells[metric.Quality]}
		return moved
	}
	// A baseline that measured the first model under the chat template, its
	// headline, and retained raw completion beside it holds a record to both.
	raw := with(func(c *Contract) { c.Protocol = "raw-completion" })
	quality := func(value float64, contract Contract) Cell {
		return row(first, "shared", value, contract).Cells[metric.Quality]
	}
	measured := func(headline Cell, others ...Cell) Row {
		measured := row(first, "shared", 0, mmlu)
		measured.Cells[metric.Quality], measured.Heldout = headline, others
		return measured
	}
	retained := &Record{Commit: "a", Rows: []Row{measured(quality(0.8, mmlu), quality(0.75, raw)), row(second, "shared", 0.7, mmlu)}}
	for _, scenario := range []struct {
		name     string
		record   Record
		verdict  Verdict
		baseline *Record
	}{
		{name: "renamed and reordered", record: record(row(second, "renamed", 0.7, mmlu), row(first, "shared", 0.5, mmlu)), verdict: holding},
		{name: "gain", record: record(row(first, "shared", 0.6, mmlu), row(second, "shared", 0.7, mmlu)), verdict: holding},
		{name: "fall on the same contract", record: record(row(first, "shared", 0.4, mmlu), row(second, "shared", 0.7, mmlu)), verdict: regressed},
		{name: "one case in a million lost", record: record(row(first, "shared", 0.5-1e-6, mmlu), row(second, "shared", 0.7, mmlu)), verdict: regressed},
		{name: "a summation-order difference", record: record(row(first, "shared", 0.5*(1-1e-8), mmlu), row(second, "shared", 0.7, mmlu)), verdict: holding},
		{name: "a larger headline suite, the baseline's suite holding", record: record(headlineMoved(0.5), row(second, "shared", 0.7, mmlu)), verdict: holding},
		{name: "a larger headline suite, the baseline's suite fallen", record: record(headlineMoved(0.4), row(second, "shared", 0.7, mmlu)), verdict: regressed},
		{name: "another suite", record: record(row(first, "shared", 0.6, with(func(c *Contract) { c.Suite = "store/mmlu-pro" })), row(second, "shared", 0.7, mmlu)), verdict: Unknown},
		{name: "another protocol", record: record(row(first, "shared", 0.6, with(func(c *Contract) { c.Protocol = "raw-completion" })), row(second, "shared", 0.7, mmlu)), verdict: Unknown},
		{name: "another origin", record: record(row(first, "shared", 0.6, with(func(c *Contract) { c.Origin = originBenchmark })), row(second, "shared", 0.7, mmlu)), verdict: Unknown},
		{name: "another direction", record: record(minimized, row(second, "shared", 0.7, mmlu)), verdict: Unknown},
		{name: "a lost cell", record: record(lost, row(second, "shared", 0.7, mmlu)), verdict: Unknown},
		{name: "a vanished row", record: record(row(second, "shared", 0.7, mmlu)), verdict: Unknown},
		{name: "every row vanished", record: record(), verdict: Unknown},
		{name: "a new row and a newly measured metric", record: record(newlyMeasured, row(second, "shared", 0.7, mmlu), row(added, "added", 0.1, mmlu)), verdict: holding},
		{name: "a retained contract holding beside the headline", baseline: retained, record: record(measured(quality(0.8, mmlu), quality(0.75, raw)), row(second, "shared", 0.7, mmlu)), verdict: holding},
		{name: "the headline and a retained contract reordered", baseline: retained, record: record(measured(quality(0.75, raw), quality(0.8, mmlu)), row(second, "shared", 0.7, mmlu)), verdict: holding},
		{name: "a new optional suite beside the baseline's", baseline: retained, record: record(measured(quality(0.1, larger), quality(0.8, mmlu), quality(0.75, raw)), row(second, "shared", 0.7, mmlu)), verdict: holding},
		{name: "a retained contract fallen while the headline holds", baseline: retained, record: record(measured(quality(0.8, mmlu), quality(0.6, raw)), row(second, "shared", 0.7, mmlu)), verdict: regressed},
		{name: "a retained contract fallen as the new headline", baseline: retained, record: record(measured(quality(0.6, raw), quality(0.8, mmlu)), row(second, "shared", 0.7, mmlu)), verdict: regressed},
		{name: "a retained contract vanished while the headline holds", baseline: retained, record: record(measured(quality(0.8, mmlu)), row(second, "shared", 0.7, mmlu)), verdict: Unknown},
	} {
		against := cmp.Or(scenario.baseline, &baseline)
		verdict := holding
		for _, pair := range scenario.record.Pairs(*against) {
			for _, column := range metric.Columns {
				for _, compared := range pair.Compare(column) {
					if compared.Why == "" {
						t.Fatalf("%s: %s %s is %s without a reason", scenario.name, pair.Before.Name, column, compared.Verdict)
					}
					verdict = compared.Verdict
				}
			}
		}
		if verdict != scenario.verdict {
			t.Errorf("%s: verdict %s, want %s", scenario.name, verdict, scenario.verdict)
			continue
		}
		withheld := scenario.record.Withheld(against)
		if (withheld == "") != (verdict == holding) {
			t.Fatalf("%s: withheld %q for a %s comparison", scenario.name, withheld, verdict)
		}
		proposed := false
		for _, gap := range scenario.record.Gaps(against) {
			proposed = proposed || gap.kind == string(verdict)
		}
		if verdict != holding && !proposed {
			t.Fatalf("%s: gaps %+v; a %s comparison is not proposed", scenario.name, scenario.record.Gaps(against), verdict)
		}
	}
	legacy := row(first, "shared", 0.5, Contract{})
	if compared := (Pair{Before: &legacy, After: &baseline.Rows[0]}).Compare(metric.Quality); len(compared) != 1 || compared[0].Verdict != Unknown || !strings.Contains(compared[0].Why, "without a contract") {
		t.Fatalf("a baseline cell published before contracts compares as %+v, want unknown", compared)
	}
	served := func(scope artifact.ID, contract func(*servingSummary) Cell) Row {
		return Row{Model: first, Task: recipe.TaskInference, Cells: map[metric.Name]Cell{metric.FirstOutput: contract(&servingSummary{latest: source, recipe: scope})}}
	}
	prefill := func(summary *servingSummary) Cell {
		return summary.cell(120, "ms", "time to prefill", "serving time to prefill")
	}
	benchmarked := func(*servingSummary) Cell { return benchmarkCell(120, "ms", "time to first token, median", source) }
	old, current := testutil.ArtifactID(t, artifact.KindRecipe, "old"), testutil.ArtifactID(t, artifact.KindRecipe, "current")
	for name, before := range map[string]Row{"another recipe's attempts": served(old, prefill), "a benchmark in the same unit": served(current, benchmarked)} {
		after := served(current, prefill)
		if compared := (Pair{Before: &before, After: &after}).Compare(metric.FirstOutput); len(compared) != 1 || compared[0].Verdict != Unknown {
			t.Fatalf("a serving median beside %s compares as %+v, want unknown", name, compared)
		}
	}
}

// TestTrainedBelowBaseIsAGap holds a trained model's held-out quality to its
// base model's from the verdict that judged them: a training row that falls
// below its base in the metric's direction, by more than float32 arithmetic
// resolves, is a gap whose line shows both values, and one that ties or
// beats its base is none. Neither withholds promotion. The base travels in
// the published record, and a record published before it still reads.
func TestTrainedBelowBaseIsAGap(t *testing.T) {
	t.Parallel()
	source := testutil.ArtifactID(t, artifact.KindEvaluation, "below-base")
	model := testutil.ArtifactID(t, artifact.KindModel, "trained")
	heldout := Contract{Metric: "token-accuracy", Origin: originEvaluation, Suite: "view", Protocol: "plan"}
	trained := func(value float64, base *float64, direction runrecord.Direction) Row {
		cells := map[metric.Name]Cell{}
		for _, column := range metric.Columns {
			cells[column] = inapplicable("not measured in this fixture")
		}
		cells[metric.Quality] = Cell{Value: value, Base: base, Label: "token-accuracy, view (plan)", Source: source, Direction: direction, Contract: heldout}
		return Row{Model: model, Name: "trained", Task: recipe.TaskTraining, Cells: cells}
	}
	of := func(value float64) *float64 { return &value }
	below := trained(0.3634, of(0.4936), runrecord.DirectionMaximize)
	const key, line = "scorecard-below-base-training", "- trained held-out-quality: 0.3634 vs base 0.4936 (token-accuracy, view (plan))\n"
	if text := below.Cells[metric.Quality].QualityText(); text != "0.3634 vs base 0.4936 (token-accuracy, view (plan))" {
		t.Fatalf("a trained model's held-out quality reads %q, want its value against its base's", text)
	}
	if text := trained(0.3634, nil, runrecord.DirectionMaximize).Cells[metric.Quality].QualityText(); text != "0.3634 (token-accuracy, view (plan))" {
		t.Fatalf("a held-out quality without a base reads %q", text)
	}
	for name, scenario := range map[string]struct {
		row  Row
		open bool
	}{
		"below its base":                    {row: below, open: true},
		"a higher perplexity than its base": {row: trained(12, of(10), runrecord.DirectionMinimize), open: true},
		"tied with its base":                {row: trained(0.4936, of(0.4936), runrecord.DirectionMaximize)},
		"a summation-order difference":      {row: trained(0.4936*(1-1e-8), of(0.4936), runrecord.DirectionMaximize)},
		"above its base":                    {row: trained(0.6, of(0.4936), runrecord.DirectionMaximize)},
		"a lower perplexity than its base":  {row: trained(8, of(10), runrecord.DirectionMinimize)},
		"no base":                           {row: trained(0.3634, nil, runrecord.DirectionMaximize)},
	} {
		record := Record{Commit: "b", Rows: []Row{scenario.row}}
		gaps := record.Gaps(nil)
		at := slices.IndexFunc(gaps, func(gap Gap) bool { return gap.Key == key })
		if (at >= 0) != scenario.open {
			t.Fatalf("%s: gaps %+v, want %s open: %t", name, gaps, key, scenario.open)
		}
		if at >= 0 && (gaps[at].Title != "Make training beat its base model on held-out quality" || !strings.Contains(gaps[at].Detail, "vs base")) {
			t.Fatalf("%s: gap %+v does not name the work or show both values", name, gaps[at])
		}
		baseline := Record{Commit: "a", Rows: []Row{scenario.row}}
		if withheld := record.Withheld(&baseline); withheld != "" {
			t.Fatalf("%s: withheld %q; a trained model below its base is a gap, not a reason to withhold the commit", name, withheld)
		}
	}
	if gaps := (Record{Rows: []Row{below}}).Gaps(nil); len(gaps) != 1 || gaps[0].Detail != line {
		t.Fatalf("a trained model below its base proposes %+v, want one gap with the line %q", gaps, line)
	}

	directory := t.TempDir()
	older := trained(0.3634, nil, runrecord.DirectionMaximize)
	zero := trained(0.25, of(0), runrecord.DirectionMaximize)
	zero.Model = testutil.ArtifactID(t, artifact.KindModel, "zero-base")
	published := time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)
	path, err := Publish(directory, published, Record{Commit: "a", Rows: []Row{older}})
	if err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(path); err != nil || strings.Contains(string(data), `"base"`) {
		t.Fatalf("a held-out quality without a base publishes %s (%v), want no base field", data, err)
	}
	if _, err := Publish(directory, published.Add(time.Hour), Record{Commit: "b", Rows: []Row{below, zero}}); err != nil {
		t.Fatal(err)
	}
	records, err := Records(directory)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 || filepath.Base(path) != records[0].Name {
		t.Fatalf("read %d records, want the older one first", len(records))
	}
	if cell := records[0].Rows[0].Cells[metric.Quality]; cell.Base != nil || cell.Value != 0.3634 {
		t.Fatalf("a record published before bases reads %+v", cell)
	}
	for index, want := range []float64{0.4936, 0} {
		if cell := records[1].Rows[index].Cells[metric.Quality]; cell.Base == nil || *cell.Base != want {
			t.Fatalf("row %d reads %+v, want its base %g back", index, cell, want)
		}
	}
	if gaps := records[1].Gaps(&records[0]); !slices.ContainsFunc(gaps, func(gap Gap) bool { return gap.Key == key && gap.Detail == line }) {
		t.Fatalf("the published record proposes %+v, want %s", gaps, key)
	}
}
