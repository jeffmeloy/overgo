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
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestComparisonRequiresEvaluationAuthorities prevents unknown measurements
// from earning a holding verdict merely because their display labels match.
// Known benchmark contracts compare through the ordinary row owner, while
// missing or changed authorities keep the measured values without a verdict.
func TestComparisonRequiresEvaluationAuthorities(t *testing.T) {
	t.Parallel()
	for _, origin := range []string{originEvaluation, originBenchmark} {
		unknown := Cell{Value: 0.5, Direction: runrecord.DirectionMaximize,
			Contract: Contract{Metric: "fixture", Origin: origin}}
		if verdict, why := holds(metric.Quality, unknown, unknown); verdict != Unknown || why == "" {
			t.Fatalf("unknown %s compares as %q (%s)", origin, verdict, why)
		}
	}
	source := testutil.ArtifactID(t, artifact.KindEvidence, "benchmark-before")
	comparison := testutil.ArtifactID(t, artifact.KindProfile, "benchmark-execution")
	baseline := benchmarkCell(40, "tok/s", "decode rate, median", evaluation.BenchmarkSummary{Record: source, Comparison: comparison})
	for _, scenario := range []struct {
		name       string
		comparison artifact.ID
		verdict    Verdict
	}{
		{"same execution", comparison, holding},
		{"unknown execution", artifact.ID{}, Unknown},
		{"another execution", testutil.ArtifactID(t, artifact.KindProfile, "other-execution"), Unknown},
	} {
		currentSource := testutil.ArtifactID(t, artifact.KindEvidence, scenario.name)
		current := benchmarkCell(45, "tok/s", "decode rate, median", evaluation.BenchmarkSummary{Record: currentSource, Comparison: scenario.comparison})
		before := Row{Cells: map[metric.Name]Cell{metric.Throughput: baseline}}
		after := Row{Cells: map[metric.Name]Cell{metric.Throughput: current}}
		compared := (Pair{Before: &before, After: &after}).Compare(metric.Throughput)
		if scenario.verdict == holding {
			if len(compared) != 0 {
				t.Fatalf("%s: matching measured values do not compare: %+v", scenario.name, compared)
			}
		} else if len(compared) != 1 || compared[0].Verdict != scenario.verdict || compared[0].Why == "" {
			t.Fatalf("%s: compares as %+v", scenario.name, compared)
		}
		if current.Value != 45 || current.Source != currentSource || !current.Measured() {
			t.Fatalf("%s: comparison qualification lost the measured value: %+v", scenario.name, current)
		}
	}
}

// TestComparisonIsTyped holds the one comparison to identity and typed
// contracts. Rows pair by model and task, never by display name: a renamed
// model keeps its reference, and two models sharing a name never trade
// theirs. A held-out quality compares only on the reference's metric, suite,
// protocol, origin and direction, where it holds or regresses. Every
// held-out quality the reference measured is held, the headline and each
// retained beside it, in whatever order the record lists them: one that
// falls while the headline holds is a decline, filed as a gap. A vanished
// row (every row, too), a lost cell, a changed contract and a reference cell
// published before contracts are unknown: they are reported and file nothing.
// A metric, a suite or a row the reference did not measure holds and asks
// nothing. Serving aggregates of other recipes' attempts, and a benchmark
// beside a serving median in the same unit, do not compare.
func TestComparisonIsTyped(t *testing.T) {
	t.Parallel()
	source := testutil.ArtifactID(t, artifact.KindEvaluation, "typed-comparison")
	first, second, added := testutil.ArtifactID(t, artifact.KindModel, "first"), testutil.ArtifactID(t, artifact.KindModel, "second"), testutil.ArtifactID(t, artifact.KindModel, "added")
	mmlu := Contract{Metric: "accuracy", Origin: originEvaluation, Suite: "store/mmlu", Protocol: "chat-template", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "mmlu-chat-environment")}
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
	newlyMeasured.Cells[metric.Load] = benchmarkCell(1, "ms", "load", evaluation.BenchmarkSummary{Record: source})
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
		{name: "one case in ten thousand lost", record: record(row(first, "shared", 0.5-1e-4, mmlu), row(second, "shared", 0.7, mmlu)), verdict: regressed},
		{name: "a summation-order difference", record: record(row(first, "shared", 0.5*(1-1e-8), mmlu), row(second, "shared", 0.7, mmlu)), verdict: holding},
		{name: "a larger headline suite, the baseline's suite holding", record: record(headlineMoved(0.5), row(second, "shared", 0.7, mmlu)), verdict: holding},
		{name: "a larger headline suite, the baseline's suite fallen", record: record(headlineMoved(0.4), row(second, "shared", 0.7, mmlu)), verdict: regressed},
		{name: "another suite", record: record(row(first, "shared", 0.6, with(func(c *Contract) { c.Suite = "store/mmlu-pro" })), row(second, "shared", 0.7, mmlu)), verdict: Unknown},
		{name: "another protocol", record: record(row(first, "shared", 0.6, with(func(c *Contract) { c.Protocol = "raw-completion" })), row(second, "shared", 0.7, mmlu)), verdict: Unknown},
		{name: "another bound contract or environment with the same label", record: record(row(first, "shared", 0.6, with(func(c *Contract) {
			c.Comparison = testutil.ArtifactID(t, artifact.KindProfile, "other-bound-environment")
		})), row(second, "shared", 0.7, mmlu)), verdict: Unknown},
		{name: "missing bound contract", record: record(row(first, "shared", 0.6, with(func(c *Contract) { c.Comparison = artifact.ID{} })), row(second, "shared", 0.7, mmlu)), verdict: Unknown},
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
		// Only a decline asks for work: an unknown is reported and files no row.
		proposed := false
		for _, gap := range scenario.record.Gaps(against) {
			proposed = proposed || gap.kind == string(regressed)
		}
		if proposed != (verdict == regressed) {
			t.Fatalf("%s: gaps %+v; a %s comparison is proposed: %t", scenario.name, scenario.record.Gaps(against), verdict, proposed)
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
		return summary.cell(summary.latest, 120, "ms", "time to prefill", "serving time to prefill")
	}
	benchmarked := func(*servingSummary) Cell {
		return benchmarkCell(120, "ms", "time to first token, median", evaluation.BenchmarkSummary{Record: source})
	}
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
	heldout := Contract{Metric: "token-accuracy", Origin: originEvaluation, Suite: "view", Protocol: "plan", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "heldout-plan-environment")}
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
		// It judges a training against its base model: held against its own earlier value, it files no decline.
		reference := Record{Commit: "a", Rows: []Row{scenario.row}}
		for _, gap := range record.Gaps(&reference) {
			if gap.kind == string(regressed) {
				t.Fatalf("%s: %+v; a trained model below its base is not a decline from itself", name, gap)
			}
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

// TestKnownGapKeyNamesTheKeysGapsProposes holds the verifier's key check to
// the keys the scorecard proposes: each kind on a task with a hyphen in its
// name is known, so is the decline of a cell (the prefix and sixteen hex digits
// of its digest), and a key missing its kind, its task or the prefix is not.
func TestKnownGapKeyNamesTheKeysGapsProposes(t *testing.T) {
	t.Parallel()
	for _, kind := range gapKinds {
		for _, task := range []recipe.Task{recipe.TaskInference, recipe.TaskImageGen, recipe.TaskVQA} {
			if key := gapKey(kind, task); !KnownGapKey(key) {
				t.Fatalf("the key %q the scorecard proposes is not known", key)
			}
		}
	}
	decline := regressionKey(Row{Model: testutil.ArtifactID(t, artifact.KindModel, "declined"), Task: recipe.TaskInference}, metric.Throughput, Cell{Contract: Contract{Metric: "rate", Origin: originServing}})
	if !KnownGapKey(decline) {
		t.Fatalf("the key %q of a decline is not known", decline)
	}
	for _, key := range []string{"", "projection", "scorecard", "scorecard-unmeasured", "scorecard-unmeasured-", "scorecard-unmeasured-nonsense", "unmeasured-inference", "scorecard-unmeasured-inference-extra",
		"scorecard-regressed-", "scorecard-regressed-inference", "scorecard-regressed-0123456789abcdef0", "scorecard-unknown-inference"} {
		if KnownGapKey(key) {
			t.Fatalf("the key %q names no gap and is known", key)
		}
	}
}

// measuredFixture is a model's row with a cell for every column, a gap naming
// its reason where the fixture measures none.
func measuredFixture(model artifact.ID, cells map[metric.Name]Cell) Row {
	for _, column := range metric.Columns {
		if _, found := cells[column]; !found {
			cells[column] = gap("no record measures it")
		}
	}
	return Row{Model: model, Name: "served", Task: recipe.TaskInference, Cells: cells}
}

// TestEarlierIsEachCellsNewestEarlierMeasurement holds what a record is
// compared with to the records before it: the newest earlier record's cells, one it
// did not measure taken from the newest older record that did, and nothing in place
// of a value the newest record lacks. A record alone has no reference, and building
// one changes no record.
func TestEarlierIsEachCellsNewestEarlierMeasurement(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "served")
	contract := Contract{Metric: "rate", Origin: originServing}
	measured := func(value float64) Cell { return Cell{Value: value, Unit: "tok/s", Contract: contract} }
	older := Record{Commit: "a", Rows: []Row{measuredFixture(model, map[metric.Name]Cell{metric.Throughput: measured(40), metric.Load: measured(9)})}}
	incomplete := Record{Commit: "b", Rows: []Row{measuredFixture(model, map[metric.Name]Cell{metric.Throughput: gap("the run failed"), metric.Load: measured(8)})}}
	current := Record{Commit: "c", Rows: []Row{measuredFixture(model, map[metric.Name]Cell{metric.Throughput: measured(25), metric.Load: measured(8)})}}
	records := []Record{older, incomplete, current}

	reference := Earlier(records)
	if reference == nil || reference.Commit != "b" {
		t.Fatalf("reference = %+v, want the newest earlier record's", reference)
	}
	if cells := reference.Rows[0].Cells; cells[metric.Throughput].Value != 40 || cells[metric.Load].Value != 8 {
		t.Fatalf("reference throughput %v and load %v, want the older record's 40 for the cell the incomplete record lacks and its own 8", cells[metric.Throughput].Value, cells[metric.Load].Value)
	}
	if incomplete.Rows[0].Cells[metric.Throughput].Measured() {
		t.Fatal("building the reference filled the incomplete record's own cell")
	}
	if Earlier(records[:1]) != nil || Earlier(nil) != nil || Reference(records[:1]) != nil || Reference(nil) != nil {
		t.Fatal("a record with none before it has a reference")
	}

	// The cell declined from the older record's value, and the incomplete record is no reference for it.
	var declined []Gap
	for _, gap := range current.Gaps(reference) {
		if gap.kind == string(regressed) {
			declined = append(declined, gap)
		}
	}
	if len(declined) != 1 || declined[0].Target != 40 || !declined[0].Restore || !strings.Contains(declined[0].Detail, "-37.5%") {
		t.Fatalf("declines = %+v, want throughput's one, to be restored to 40, with its raw change", declined)
	}

	// A value the current record lacks is Unknown, never answered by an older one, and files no decline.
	lost := Record{Commit: "d", Rows: []Row{measuredFixture(model, map[metric.Name]Cell{metric.Throughput: gap("the run failed"), metric.Load: measured(8)})}}
	pair := Pair{Before: &reference.Rows[0], After: &lost.Rows[0]}
	if compared := pair.Compare(metric.Throughput); len(compared) != 1 || compared[0].Verdict != Unknown || !strings.Contains(compared[0].Why, "no current value: the run failed") {
		t.Fatalf("a cell with no current value compares as %+v, want unknown", compared)
	}
	for _, gap := range lost.Gaps(reference) {
		if gap.kind == string(regressed) {
			t.Fatalf("a cell with no current value is filed as a decline: %+v", gap)
		}
	}
}

// TestADeclineMustPassWhatItsReferenceAllows holds each column to what a decline is in it. A
// timing or a peak memory declines by more than its reference allows, each side of the line,
// and is Unknown against a reference whose readings are too few. A quality, a count of setup
// steps and a memory whose reference allows nothing decline by any move the scorecard's four
// printed digits show and by none they do not: one case lost of ten thousand is a decline, a
// perplexity that moved by parts in a billion and a peak memory a few hundred bytes larger
// are not. A failed share declines by more than one failure beyond what the earlier share
// predicts at the attempts counted now, and not at all when either share lost its attempts.
func TestADeclineMustPassWhatItsReferenceAllows(t *testing.T) {
	t.Parallel()
	contract := Contract{Metric: "rate", Origin: originServing}
	cell := func(value float64) Cell { return Cell{Value: value, Contract: contract} }
	allowing := func(value, allowed float64) Cell { return Cell{Value: value, Contract: contract, allowed: allowed} }
	failed := func(failures, attempts int) Cell {
		return Cell{Value: float64(failures) / float64(attempts), Attempts: attempts, Contract: contract}
	}
	setup := func(steps float64) Cell {
		return Cell{Value: steps, Contract: Contract{Metric: "transitions", Origin: originLifecycle}}
	}
	scored := func(value float64, direction runrecord.Direction) Cell {
		return Cell{Value: value, Direction: direction, Contract: contract}
	}
	few := cell(100)
	few.unjudged = "three earlier readings"
	for _, test := range []struct {
		name         string
		column       metric.Name
		then, now    Cell
		wantVerdict  Verdict
		wantContains string
	}{
		{"first output inside what its reference allows", metric.FirstOutput, allowing(100, 35), cell(134), holding, ""},
		{"first output beyond it", metric.FirstOutput, allowing(100, 35), cell(160), regressed, "+60.0%, allowed 35;"},
		{"first output faster", metric.FirstOutput, allowing(100, 35), cell(40), holding, ""},
		{"first output against too few readings", metric.FirstOutput, few, cell(160), Unknown, "three earlier readings"},
		{"throughput inside what its reference allows", metric.Throughput, allowing(40, 6), cell(35), holding, ""},
		{"throughput beyond it", metric.Throughput, allowing(40, 6), cell(33), regressed, "-17.5%"},
		{"throughput faster", metric.Throughput, allowing(40, 6), cell(80), holding, ""},
		{"load inside what its reference allows", metric.Load, allowing(5000, 2500), cell(7000), holding, ""},
		{"load beyond it", metric.Load, allowing(5000, 2500), cell(8000), regressed, ""},
		{"peak memory a few hundred bytes more", metric.PeakMemory, cell(5650920830), cell(5650921340), holding, ""},
		{"peak memory a printed digit more", metric.PeakMemory, cell(5650920830), cell(5652000000), regressed, "allowed 500000;"},
		{"a quality one case in ten thousand lower", metric.Quality, scored(0.2855, runrecord.DirectionMaximize), scored(0.2854, runrecord.DirectionMaximize), regressed, ""},
		{"a full quality one case in ten thousand lower", metric.Quality, scored(1, runrecord.DirectionMaximize), scored(0.9999, runrecord.DirectionMaximize), regressed, ""},
		{"a quality a summation order lower", metric.Quality, scored(0.2855, runrecord.DirectionMaximize), scored(0.2855*(1-1e-8), runrecord.DirectionMaximize), holding, ""},
		{"a perplexity parts in a billion higher", metric.Quality, scored(910.321, runrecord.DirectionMinimize), scored(910.321+6.5e-6, runrecord.DirectionMinimize), holding, ""},
		{"a perplexity a printed digit higher", metric.Quality, scored(910.321, runrecord.DirectionMinimize), scored(910.4, runrecord.DirectionMinimize), regressed, ""},
		{"setup steps the same", metric.SetupSteps, setup(3), setup(3), holding, ""},
		{"setup steps one more", metric.SetupSteps, setup(3), setup(4), regressed, "+1"},
		{"setup steps fewer", metric.SetupSteps, setup(4), setup(3), holding, ""},
		{"one more failure in a hundred", metric.Failures, failed(1, 100), failed(2, 100), holding, ""},
		{"two more failures in a hundred", metric.Failures, failed(1, 100), failed(3, 100), regressed, ""},
		{"the same share over ten times the attempts", metric.Failures, failed(1, 100), failed(10, 1000), holding, ""},
		{"one failure in ten against a hundredth", metric.Failures, failed(1, 100), failed(1, 10), holding, ""},
		{"three failures in ten against a hundredth", metric.Failures, failed(1, 100), failed(3, 10), regressed, ""},
		{"a share published without its attempts", metric.Failures, cell(0.01), failed(2, 100), Unknown, "without the attempts"},
	} {
		verdict, why := holds(test.column, test.then, test.now)
		if verdict != test.wantVerdict || !strings.Contains(why, test.wantContains) {
			t.Errorf("%s: holds = %q (%s), want %q containing %q", test.name, verdict, why, test.wantVerdict, test.wantContains)
		}
	}
}

// TestADeclineKeepsItsKeyAndItsOriginalRestorationTarget holds a decline to the one cell
// that declined: its key names the model, the task, the measure and the whole contract and
// no value, so a cell that stays down, or goes further down, is found again under the same
// key, and the target its row carries is the value it declined from. Of two held-out suites,
// the one that fell is the one filed. A decline is restored when the cell is back within
// the column's noise of the target, and a key no cell carries is not.
func TestADeclineKeepsItsKeyAndItsOriginalRestorationTarget(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "declined")
	source := testutil.ArtifactID(t, artifact.KindEvaluation, "decline")
	mmlu := Contract{Metric: "accuracy", Origin: originEvaluation, Suite: "store/mmlu", Protocol: "chat-template", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "mmlu-chat-environment")}
	pro := mmlu
	pro.Suite = "store/mmlu-pro"
	quality := func(value float64, contract Contract) Cell {
		return Cell{Value: value, Label: "accuracy", Source: source, Direction: runrecord.DirectionMaximize, Contract: contract}
	}
	row := func(headline, other float64) Row {
		fixture := measuredFixture(model, map[metric.Name]Cell{metric.Quality: quality(headline, mmlu)})
		fixture.Heldout = []Cell{quality(other, pro)}
		return fixture
	}
	declines := func(now, reference Row) []Gap {
		var found []Gap
		for _, gap := range (Record{Commit: "b", Rows: []Row{now}}).Gaps(&Record{Commit: "a", Rows: []Row{reference}}) {
			if gap.kind == string(regressed) {
				found = append(found, gap)
			}
		}
		return found
	}
	first := declines(row(0.8, 0.5), row(0.8, 0.7))
	if len(first) != 1 || first[0].Target != 0.7 || !first[0].Restore || !strings.Contains(first[0].Detail, "store/mmlu-pro") || strings.Contains(first[0].Detail, "mmlu,") {
		t.Fatalf("declines = %+v, want only the suite that fell, to be restored to 0.7", first)
	}
	// It stays down, then goes further down: the same key, whatever it is compared with.
	again := declines(row(0.8, 0.4), row(0.8, 0.5))
	if len(again) != 1 || again[0].Key != first[0].Key || again[0].Target != 0.5 {
		t.Fatalf("a further decline = %+v, want the key %s", again, first[0].Key)
	}
	if got := declines(row(0.8, 0.5), row(0.8, 0.5)); len(got) != 0 {
		t.Fatalf("a cell that stayed down against itself is a decline again: %+v", got)
	}
	// The other suite declining is another cell, so another key.
	if other := declines(row(0.6, 0.7), row(0.8, 0.7)); len(other) != 1 || other[0].Key == first[0].Key {
		t.Fatalf("the headline suite declining = %+v, want a key of its own", other)
	}

	rows := func(headline, other float64) []Row { return []Row{row(headline, other)} }
	for name, test := range map[string]struct {
		rows            []Row
		restored, found bool
	}{
		"still down":             {rows(0.8, 0.5), false, true},
		"back":                   {rows(0.8, 0.7), true, true},
		"above where it was":     {rows(0.8, 0.9), true, true},
		"within the noise of it": {rows(0.8, 0.7*(1-1e-8)), true, true},
		"another model's cell":   {[]Row{{Model: testutil.ArtifactID(t, artifact.KindModel, "other"), Task: recipe.TaskInference, Cells: row(0.8, 0.7).Cells, Heldout: row(0.8, 0.7).Heldout}}, false, false},
	} {
		if restored, found := Restored(test.rows, first[0].Key, 0.7); restored != test.restored || found != test.found {
			t.Errorf("%s: Restored = (%t, %t), want (%t, %t)", name, restored, found, test.restored, test.found)
		}
	}
}
