package scorecard

import (
	"strings"
	"testing"

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
// protocol, origin and direction, where it holds or regresses. A vanished
// row (every row, too), a lost cell, a changed contract and a baseline cell
// published before contracts are unknown: they withhold promotion and are
// proposed as a gap. A metric or a row the baseline did not measure holds
// and blocks nothing. Serving aggregates of other recipes' attempts, and a
// benchmark beside a serving median in the same unit, do not compare.
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
	for _, scenario := range []struct {
		name    string
		record  Record
		verdict Verdict
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
	} {
		verdict := holding
		for _, pair := range scenario.record.Pairs(baseline) {
			for _, column := range metric.Columns {
				if got, why := pair.Compare(column); got != holding {
					if why == "" {
						t.Fatalf("%s: %s %s is %s without a reason", scenario.name, pair.Before.Name, column, got)
					}
					verdict = got
				}
			}
		}
		if verdict != scenario.verdict {
			t.Fatalf("%s: verdict %s, want %s", scenario.name, verdict, scenario.verdict)
		}
		withheld := scenario.record.Withheld(&baseline)
		if (withheld == "") != (verdict == holding) {
			t.Fatalf("%s: withheld %q for a %s comparison", scenario.name, withheld, verdict)
		}
		proposed := false
		for _, gap := range scenario.record.Gaps(&baseline) {
			proposed = proposed || gap.kind == string(verdict)
		}
		if verdict != holding && !proposed {
			t.Fatalf("%s: gaps %+v; a %s comparison is not proposed", scenario.name, scenario.record.Gaps(&baseline), verdict)
		}
	}
	legacy := row(first, "shared", 0.5, Contract{})
	if verdict, why := (Pair{Before: &legacy, After: &baseline.Rows[0]}).Compare(metric.Quality); verdict != Unknown || !strings.Contains(why, "without a contract") {
		t.Fatalf("a baseline cell published before contracts compares as %s (%q), want unknown", verdict, why)
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
		if verdict, _ := (Pair{Before: &before, After: &after}).Compare(metric.FirstOutput); verdict != Unknown {
			t.Fatalf("a serving median beside %s compares as %s, want unknown", name, verdict)
		}
	}
}
